import json, re
from store import collection

SKIP = {
    "extracted_images\\Python Notes_p1_0.png",
    "extracted_images\\Python Notes_p1_2.png",
}

before = collection.count()
recs = json.load(open("images_manifest.json"))
ids, docs, metas = [], [], []
for r in recs:
    if not r.get("description") or r["image_path"] in SKIP:
        continue
    desc = re.sub(r"^Based on the image provided,?\s*here is the description:\s*", "", r["description"]).strip()
    ids.append("img_" + r["image_path"].replace("\\", "_").replace("/", "_"))
    docs.append(desc)
    metas.append({
        "source_type": "image",
        "source_name": r["source_name"],
        "page": r["page"],`n        "page_number": r["page"],
        "image_path": r["image_path"],
    })

collection.upsert(ids=ids, documents=docs, metadatas=metas)
print("pehle:", before, "baad mein:", collection.count(), "| image chunks:", len(ids))
