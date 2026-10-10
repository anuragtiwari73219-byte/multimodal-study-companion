import fitz
import os, sys, json

MIN_W, MIN_H = 150, 150

def extract(pdf_path, out_dir="extracted_images"):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    records = []
    for page_num, page in enumerate(doc, start=1):
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.width < MIN_W or pix.height < MIN_H:
                continue
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fname = f"{base}_p{page_num}_{idx}.png"
            fpath = os.path.join(out_dir, fname)
            pix.save(fpath)
            records.append({"source_name": os.path.basename(pdf_path),
                            "page": page_num, "image_path": fpath,
                            "width": pix.width, "height": pix.height})
    return records

if __name__ == "__main__":
    recs = extract(sys.argv[1])
    print(f"{len(recs)} images nikli")
    for r in recs[:5]:
        print(r)
    json.dump(recs, open("images_manifest.json", "w"), indent=2)
