import json
kb = json.load(open('backend/data/waste_knowledge_base.json', encoding='utf-8'))
for item in kb['items']:
    print(f"{item['class_id']:<38} | {item['primary_stream']:<20} | {item['subcategory']}")
