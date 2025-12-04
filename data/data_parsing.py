import numpy
import os

def parse_wider_face(txt_path, root_img_dir): 
    image_paths = []
    targets = []
    current_path = None
    current_boxes = []

    if not os.path.exists(txt_path): 
        raise FileNotFoundError(f"File label tidak ditemukan: {txt_path}")
    print(f"Sedang membaca anotasi dari: {txt_path} ...")

    with open(txt_path, 'r') as f: 
        lines = f.readlines()
        
    for line in lines: 
        line = line.strip()
        if not line: 
            continue
            
        if line.startswith('#'): 
            # --- BAGIAN HEADER GAMBAR ---
            if current_path is not None: 
                image_paths.append(current_path)
                if len(current_boxes) > 0: 
                    targets.append(numpy.array(current_boxes, dtype=numpy.float32))
                else: 
                    targets.append(numpy.zeros((0, 4), dtype=numpy.float32))
            
            parts = line.split()
            # Pastikan format label Anda benar. Biasanya index 1 itu pathnya.
            # Contoh: # 0--Parade/0_Parade... 
            if len(parts) > 1:
                rel_path = parts[1] 
            else:
                rel_path = parts[0].replace('#', '').strip() # Jaga-jaga format beda

            current_path = os.path.join(root_img_dir, rel_path)
            current_boxes = []
            
        else: 
            # --- BAGIAN KOORDINAT BOX ---
            coords = line.split()
            box = [float(x) for x in coords[:4]]
            kps = [float(y) for y in coords[4:]]
        
            # <<< PERBAIKAN DISINI >>>
            # Logika ini HARUS sejajar (indent) di dalam else
            # agar hanya dijalankan saat 'box' sudah terdefinisi.
            if box[2] > box[0] and box[3] > box[1]:
                current_boxes.append(box)

    # Jangan lupa simpan gambar terakhir setelah loop selesai
    if current_path is not None: 
        image_paths.append(current_path)
        if len(current_boxes) > 0: 
            targets.append(numpy.array(current_boxes, dtype=numpy.float32))
        else: 
            targets.append(numpy.zeros((0, 4), dtype=numpy.float32))
            
    print(f"Selesai parsing. Ditemukan {len(image_paths)} gambar.")
    return image_paths, targets

if __name__=="__main__": 
    # Pastikan path ini benar ada di komputer Anda
    label_file = "data/WIDER_train/labelv2_test.txt"
    image_root = "data/WIDER_train/images"

    try: 
        paths, boxes = parse_wider_face(label_file, image_root)

        if len(paths) > 0:
            print("\nSampel Data Pertama:")
            print("Path:", paths[0]) # Print index 0 saja biar terminal ga penuh
            print("Boxes (x1, y1, x2, y2):")
            print(boxes[0])
        else:
            print("Data kosong! Cek isi file txt.")

    except Exception as e: 
        print(f"Error: {e}")