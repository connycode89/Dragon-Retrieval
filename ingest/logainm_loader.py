"""
Logainm placename loader.

Fetches Irish placename records from the Logainm API (v1.0) and converts
them to LangChain Documents with etymology and linguistic metadata.

API docs: https://docs.gaois.ie/en/data/logainm/v1.0/api
API key:  register at gaois.ie or email logainm@dcu.ie
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document
from config import LOGAINM_API_BASE, LOGAINM_API_KEY


class LogainmLoader:
    """Load placename documents from the Logainm API."""

    def __init__(self, api_key: str = LOGAINM_API_KEY):
        if not api_key:
            raise ValueError(
                "Logainm API key required. Register at gaois.ie or email logainm@dcu.ie, "
                "then set LOGAINM_API_KEY in your .env file."
            )
        self.api_key = api_key
        self.headers = {"X-Api-Key": api_key}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _get(self, endpoint: str = "", params: dict = None) -> dict:
        url = f"{LOGAINM_API_BASE}/{endpoint}".rstrip("/")
        response = httpx.get(url, headers=self.headers, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()

    def load_by_place_id(self, place_id: int) -> Document | None:
        """Fetch a single place record by its Logainm ID."""
        data = self._get(str(place_id))
        return self._to_document(data) if data else None

    def load_by_county(self, county_id: int, category: str = "PAR", per_page: int = 200) -> list[Document]:
        """
        Fetch placenames within a county.

        category: 'PAR' = parish, 'BAR' = barony, 'TLD' = townland, etc.
        """
        params = {
            "PlaceID": county_id,
            "CategoryID": category,
            "PerPage": per_page,
        }
        data = self._get(params=params)
        places = data if isinstance(data, list) else data.get("Places", [])
        return [doc for p in places if (doc := self._to_document(p)) is not None]

    def search(self, query: str, max_results: int = 20) -> list[Document]:
        """Free-text search for a placename."""
        params = {"Query": query, "PerPage": max_results}
        data = self._get(params=params)
        places = data if isinstance(data, list) else data.get("Places", [])
        return [doc for p in places if (doc := self._to_document(p)) is not None]

    def load_by_coordinates(self, lat: float, lon: float, radius_m: int = 5000) -> list[Document]:
        """Fetch all named places within a radius of a coordinate point."""
        params = {
            "Latitude": lat,
            "Longitude": lon,
            "Radius": radius_m,
        }
        data = self._get(params=params)
        places = data if isinstance(data, list) else data.get("Places", [])
        return [doc for p in places if (doc := self._to_document(p)) is not None]

    def _to_document(self, place: dict) -> Document | None:
        """Convert a Logainm place record to a LangChain Document."""
        if not place:
            return None

        name_en = place.get("NameEN") or place.get("NameHI") or "Unknown"
        name_ga = place.get("NameGA") or ""
        place_id = place.get("ID", "")
        category = place.get("Category", {}) or {}
        category_name = category.get("NameEN", "") if isinstance(category, dict) else ""

        parts = [f"Place: {name_en}"]
        if name_ga:
            parts.append(f"Irish name: {name_ga}")
        if category_name:
            parts.append(f"Type: {category_name}")

        # Etymology / name elements
        glossary_entries = place.get("GlossaryEntries", []) or []
        if glossary_entries:
            parts.append("Name elements:")
            for entry in glossary_entries:
                if isinstance(entry, dict):
                    elem = entry.get("Element", {}) or {}
                    meaning = elem.get("DefinitionEN", "") if isinstance(elem, dict) else ""
                    form = entry.get("Form", "")
                    if meaning:
                        parts.append(f"  - '{form}': {meaning}")

        # Alternative name forms
        alt_names = place.get("AltNames", []) or []
        if alt_names:
            forms = [a.get("Name", "") for a in alt_names if isinstance(a, dict) and a.get("Name")]
            if forms:
                parts.append(f"Historical forms: {', '.join(forms)}")

        # Administrative context
        counties = place.get("Counties", []) or []
        if counties and isinstance(counties[0], dict):
            county_name = counties[0].get("NameEN", "")
            if county_name:
                parts.append(f"County: {county_name}")

        text = "\n".join(parts)

        metadata = {
            "source": "logainm",
            "place_id": place_id,
            "name_en": name_en,
            "name_ga": name_ga,
            "category": category_name,
            "county": counties[0].get("NameEN", "") if counties and isinstance(counties[0], dict) else "",
        }

        return Document(page_content=text, metadata=metadata)
