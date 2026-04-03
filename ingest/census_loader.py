"""
Census / historical demographics loader.

The 1901/1911 Irish Census has no public API or bulk download from the
National Archives. This module uses the best available free alternatives:

  1. Wikidata SPARQL — historical population figures for named Irish places
     (free, no auth, query via https://query.wikidata.org)
  2. data.gov.ie CKAN API — aggregate historical datasets and townland data
     (free, no auth, https://data.gov.ie/pages/developers)
  3. Wikipedia demographic articles — rich narrative context

For full individual-level 1901/1911 microdata (4.4M person records each),
register at IPUMS International: https://international.ipums.org/international/
They require a free account + signed educational-use license, and offer
CSV exports. This is out of scope for automated ingestion but worth knowing.

Note: The 1926 Census (first Free State census) was released April 2026
at nationalarchives.ie — no API yet but web-searchable.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document
from config import WIKIPEDIA_API_BASE

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
DATA_GOV_IE_API = "https://data.gov.ie/api/3/action"

# Wikipedia articles with reliable Irish historical demographic content
CENSUS_WIKI_ARTICLES = [
    "Census in Ireland",
    "Demographics of the Republic of Ireland",
    "Population history of Ireland",
    "Gaeltacht",
    "Irish language",
    "Great Famine (Ireland)",
    "Irish diaspora",
    "Townland",
]

# Wikidata SPARQL: Irish places with historical population data
# Fetches places with a population statement that has a point-in-time qualifier
WIKIDATA_POPULATION_QUERY = """
SELECT ?place ?placeLabel ?population ?pointInTime ?countyLabel WHERE {
  ?place wdt:P17 wd:Q22890 .          # located in Ireland
  ?place p:P1082 ?populationStatement .
  ?populationStatement ps:P1082 ?population .
  OPTIONAL { ?populationStatement pq:P585 ?pointInTime . }
  OPTIONAL { ?place wdt:P131 ?county .
             ?county wdt:P31 wd:Q179872 . }  # county of Ireland
  FILTER(?population > 0)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
ORDER BY ?placeLabel
LIMIT 2000
"""

# data.gov.ie CKAN search terms likely to return historical datasets
DATA_GOV_SEARCH_TERMS = ["census townland", "historical population ireland", "griffiths valuation"]


class CensusLoader:
    """
    Load historical demographic context from free public sources.

    Sources (all free, no API key required):
    - Wikidata SPARQL for historical population figures on named places
    - data.gov.ie CKAN API for aggregate datasets
    - Wikipedia for narrative demographic context
    """

    def __init__(self):
        self.client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "DragonRetrieval/1.0 (educational project)"},
        )

    def load_all(self) -> list[Document]:
        """Load all available demographic context documents."""
        docs = []
        docs.extend(self._load_wikidata_populations())
        docs.extend(self._load_data_gov_ie())
        docs.extend(self._load_wikipedia_demographics())
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _load_wikidata_populations(self) -> list[Document]:
        """
        Query Wikidata SPARQL for historical population figures of Irish places.
        Groups results by place into one Document per place.
        """
        try:
            response = self.client.get(
                WIKIDATA_SPARQL_URL,
                params={"query": WIKIDATA_POPULATION_QUERY, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=45,
            )
            response.raise_for_status()
            results = response.json().get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"  [census] Wikidata SPARQL unavailable: {e}")
            return []

        # Group population records by place
        by_place: dict[str, dict] = {}
        for row in results:
            label = row.get("placeLabel", {}).get("value", "")
            pop = row.get("population", {}).get("value", "")
            year = row.get("pointInTime", {}).get("value", "")[:4] if row.get("pointInTime") else "unknown"
            county = row.get("countyLabel", {}).get("value", "")

            if not label or label.startswith("Q"):  # skip unlabelled Wikidata items
                continue

            if label not in by_place:
                by_place[label] = {"county": county, "records": []}
            by_place[label]["records"].append(f"{year}: {pop} people")

        docs = []
        for place_name, info in by_place.items():
            lines = [f"Place: {place_name}"]
            if info["county"]:
                lines.append(f"County: {info['county']}")
            lines.append("Historical population:")
            lines.extend(f"  - {r}" for r in sorted(info["records"]))
            docs.append(Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "census_wikidata",
                    "name_en": place_name,
                    "county": info["county"],
                    "url": "https://query.wikidata.org",
                },
            ))

        print(f"  [census] Wikidata: {len(docs)} Irish places with historical population data")
        return docs

    def _load_data_gov_ie(self) -> list[Document]:
        """
        Query the data.gov.ie CKAN API for historical Irish datasets.
        Returns metadata documents describing available datasets.
        """
        docs = []
        for term in DATA_GOV_SEARCH_TERMS:
            try:
                response = self.client.get(
                    f"{DATA_GOV_IE_API}/package_search",
                    params={"q": term, "rows": 5},
                    timeout=15,
                )
                if response.status_code != 200:
                    continue
                results = response.json().get("result", {}).get("results", [])
                for dataset in results:
                    text = self._dataset_to_text(dataset)
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": "data_gov_ie",
                                "dataset_id": dataset.get("id", ""),
                                "url": f"https://data.gov.ie/dataset/{dataset.get('name', '')}",
                            },
                        ))
            except Exception:
                continue

        print(f"  [census] data.gov.ie: {len(docs)} dataset descriptions loaded")
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _load_wikipedia_demographics(self) -> list[Document]:
        """Fetch Wikipedia articles about Irish demographics for narrative context."""
        params = {
            "action": "query",
            "titles": "|".join(CENSUS_WIKI_ARTICLES),
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        try:
            response = self.client.get(WIKIPEDIA_API_BASE, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  [census] Wikipedia unavailable: {e}")
            return []

        docs = []
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("missing") is not None:
                continue
            title = page.get("title", "")
            extract = (page.get("extract") or "").strip()
            if extract and len(extract) > 100:
                docs.append(Document(
                    page_content=f"{title}\n\n{extract}",
                    metadata={
                        "source": "census_wikipedia",
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    },
                ))

        print(f"  [census] Wikipedia: {len(docs)} demographic articles loaded")
        return docs

    def _dataset_to_text(self, dataset: dict) -> str:
        """Convert a CKAN dataset record to readable text."""
        title = dataset.get("title", "")
        notes = (dataset.get("notes") or "").strip()
        tags = [t.get("name", "") for t in dataset.get("tags", [])]
        resources = [r.get("name", "") for r in dataset.get("resources", [])]

        if not title:
            return ""

        parts = [f"Dataset: {title}"]
        if notes:
            parts.append(notes[:400])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if resources:
            parts.append(f"Files: {', '.join(resources[:5])}")

        return "\n".join(parts)
