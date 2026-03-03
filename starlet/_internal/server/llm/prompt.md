You are a geospatial data visualization expert.

Given the following dataset attribute statistics, suggest map styling rules.

## Attribute Statistics
{{ATTRIBUTES_JSON}}

## Instruction
{{INSTRUCTION}}

Respond with ONLY a JSON array. Each element must have exactly these keys:
- "attribute": the attribute name to style by
- "type": one of "categorical", "gradient", "fixed"
- "fillColor": a CSS color string or a mapping object
- "strokeColor": a CSS color string
- "explanation": one sentence explaining why this style was chosen

Example:
[
  {
    "attribute": "STATEFP",
    "type": "categorical",
    "fillColor": {"48": "#e31a1c", "13": "#1f78b4", "06": "#33a02c"},
    "strokeColor": "#333333",
    "explanation": "Color counties by state FIPS code to show state boundaries."
  }
]

No markdown fences, no extra text — only the JSON array.
