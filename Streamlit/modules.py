import pandas as pd
import json
import re
 
 
# Function to extract unique model name
def extract_unique_model_name(attribute):
    # If attribute is a string, we need to load it as a dictionary
    if isinstance(attribute, str):
        parsed_data = json.loads(attribute)  # Assuming the 'attributes' column is a JSON string
    else:
        parsed_data = attribute  # Already parsed if it's already a dict
 
    # Convert to string for regex matching
    changed = json.dumps(parsed_data)
    # Regex pattern to match the model name like gpt-<version>-<suffix>
    pattern = r'gpt[-\w\.]+'  # allows dash, word chars, and dots
    # r'(\'gpt[-\w\.]+\'|"gpt[-\w\.]+")'
    matches = re.findall(pattern, changed, flags=re.IGNORECASE)
    return list(set(matches))
    # Keep only unique matches and return the first one (or adjust if you want more logic)
    #return list(set(matches))[0] if matches else None  # If matches found, return the first unique model name
 
def agentCallsPerModel(df):
    df['model_name']=df['attributes'].apply(extract_unique_model_name)

    # Explode if multiple models found
    model_df = df.explode('model_name')

    # Drop rows with no model name
    model_df = model_df[model_df['model_name'].notnull()]

    # --- Group data by model_name and agent_name ---
    grouped = model_df.groupby(['model_name', 'agent_name']).size().reset_index(name='count')

    return grouped