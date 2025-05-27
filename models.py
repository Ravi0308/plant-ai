import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Section(BaseModel):
    title: str
    items: Dict[str, str]

class PlantResponse(BaseModel):
    sections: Dict[str, Section]

def parse_plant_response(text: str) -> PlantResponse:
    sections = {}
    current_section = None
    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if line.startswith("##"):
            section_title = line.replace("##", "").strip()
            sections[section_title] = Section(title=section_title, items={})
            current_section = section_title
        elif line.startswith("•") and current_section:
            # Handle both formats: with and without **
            if "**" in line:
                key, value = line[1:].split("**", 1)
            else:
                key, value = line[1:].split(":", 1)
            sections[current_section].items[key.strip()] = value.strip()

    return PlantResponse(sections=sections)

# Example usage in your API response formatting:
def format_plant_response(response_text: str) -> str:
    structured = parse_plant_response(response_text)
    formatted_text = []

    for section_title, section in structured.sections.items():
        formatted_text.append(f"## {section_title}\n")
        for key, value in section.items.items():
            formatted_text.append(f"• {key}: {value}\n")
        formatted_text.append("\n")

    return "".join(formatted_text)

class PlantDisease(BaseModel):
    symptoms: Section = Field(..., description="Disease symptoms")
    cultural_practices: Section = Field(..., description="Cultural practices for prevention")
    treatments: Section = Field(..., description="Treatment options")
    prevention: Section = Field(..., description="Prevention methods")

class PlantIdentification(BaseModel):
    name: str = Field(..., description="Plant name")
    confidence: str = Field(..., description="Identification confidence")
    care_instructions: Section = Field(..., description="Care instructions")
    diseases: Optional[List[str]] = Field(default_factory=list, description="Common diseases")
    additional_info: Optional[str] = Field(None, description="Additional information")

class ChatResponse(BaseModel):
    sections: List[Section] = Field(..., description="Organized response sections")
    source: str = Field(default="general", description="Information source")