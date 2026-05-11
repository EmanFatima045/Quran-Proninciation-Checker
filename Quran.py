import os
import requests

# Create folder
os.makedirs("quran_audio", exist_ok=True)

# Base URL (Abdul Basit recitation)
base_url = "https://everyayah.com/data/Abdul_Basit_Murattal_192kbps/"

def format_ayah(surah, ayah):
    return f"{surah:03d}{ayah:03d}.mp3"

# Example: download first 2 surahs (for testing)
for surah in range(1, 3):  # change to 115 later
    for ayah in range(1, 300):  # safe upper bound
        filename = format_ayah(surah, ayah)
        url = base_url + filename
        
        save_path = os.path.join("quran_audio", filename)

        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded {filename}")
            else:
                break  # stop when ayahs end
        except:
            break