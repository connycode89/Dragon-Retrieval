"""
Dúchas / Schools' Collection loader.

Fetches folklore items from the Dúchas API (v0.6) and converts them
to LangChain Documents.

API docs: https://docs.gaois.ie/en/data/duchas/v0.6/api
API key:  email eolas@duchas.ie to request one
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document
from config import DUCHAS_API_BASE, DUCHAS_API_KEY


class DuchasLoader:
    """Load folklore documents from the Dúchas Schools' Collection API."""

    def __init__(self, api_key: str = DUCHAS_API_KEY):
        if not api_key:
            raise ValueError(
                "Dúchas API key required. Email eolas@duchas.ie to request one, "
                "then set DUCHAS_API_KEY in your .env file."
            )
        self.api_key = api_key
        self.headers = {"X-Api-Key": api_key}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{DUCHAS_API_BASE}/{endpoint}"
        response = httpx.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def load_by_county(self, county_id: int, max_items: int = 200) -> list[Document]:
        """Fetch folklore items for a given Dúchas county ID."""
        params = {"CountyID": county_id}
        data = self._get("cbes", params)

        docs = []
        items = data if isinstance(data, list) else data.get("data", [])

        for item in items[:max_items]:
            text = self._extract_text(item)
            if not text:
                continue

            metadata = {
                "source": "duchas",
                "county_id": county_id,
                "item_id": item.get("ID", ""),
                "school": item.get("School", {}).get("Name", "") if isinstance(item.get("School"), dict) else "",
                "county": item.get("County", {}).get("NameEN", "") if isinstance(item.get("County"), dict) else "",
                "volume_id": item.get("VolumeID", ""),
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def load_by_place(self, place_id: int, max_items: int = 100) -> list[Document]:
        """Fetch folklore items for a given Dúchas place ID."""
        params = {"PlaceID": place_id}
        data = self._get("cbes", params)

        docs = []
        items = data if isinstance(data, list) else data.get("data", [])

        for item in items[:max_items]:
            text = self._extract_text(item)
            if not text:
                continue

            metadata = {
                "source": "duchas",
                "place_id": place_id,
                "item_id": item.get("ID", ""),
                "school": item.get("School", {}).get("Name", "") if isinstance(item.get("School"), dict) else "",
                "county": item.get("County", {}).get("NameEN", "") if isinstance(item.get("County"), dict) else "",
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _extract_text(self, item: dict) -> str:
        """Pull readable text out of a Dúchas API item."""
        parts = []

        # Title / topic
        title = item.get("Title", "") or item.get("Topic", "")
        if title:
            parts.append(f"Topic: {title}")

        # Transcription pages
        pages = item.get("Pages", []) or []
        for page in pages:
            transcription = page.get("Transcription", "") or ""
            if transcription:
                parts.append(transcription.strip())

        # Flat transcription field (some items)
        if not pages:
            transcription = item.get("Transcription", "") or ""
            if transcription:
                parts.append(transcription.strip())

        # Informant details for provenance
        informant = item.get("Informant", {}) or {}
        if isinstance(informant, dict) and informant.get("Name"):
            parts.append(f"Recorded from: {informant['Name']}")
            if informant.get("AddressEN"):
                parts.append(f"Location: {informant['AddressEN']}")

        return "\n\n".join(parts)
