import urllib.request
import urllib.parse
import json

def query_wikidata(entity):
    base_url = "https://www.wikidata.org/w/api.php"
    encoded_entity = urllib.parse.quote(entity)  # Encode entity name
    query = f"?action=wbsearchentities&search={encoded_entity}&language=en&format=json"
    url = base_url + query
    
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
        return data.get("search", [])

def rank_candidates(candidates, context):
    ranked = []
    for candidate in candidates:
        description = candidate.get("description", "").lower()
        label = candidate.get("label", "").lower()
        score = 0
        # Score based on context match
        if context.lower() in description or context.lower() in label:
            score += 10
        ranked.append((candidate, score))
    # Sort by score
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [candidate for candidate, _ in ranked]

def format_linked_entity(entity):
    label = entity.get("label")
    url = entity.get("concepturi")
    return f"{label}\t{url}"

def disambiguate(entity, context):
    candidates = query_wikidata(entity)
    ranked_candidates = rank_candidates(candidates, context)
    if ranked_candidates:
        return format_linked_entity(ranked_candidates[0])
    return f"{entity}\tNo match found"

# Example usage
context = "Who is the director of Pulp Fiction?"
entities = ["Quentin Tarantino", "Reservoir Dogs"]

# Example
#context = "What is the capital of Italy?"
#entities = ["Rome"]



for entity in entities:
    print(disambiguate(entity, context))