from PIL import Image
import os, glob, re

def find_latest_imgs(img_folder="./logs/", img_name="generated_plot_e", img_ext=".png"):
    full_path = os.path.join(img_folder, img_name + "*" + img_ext)
    img_files = glob.glob(full_path)
    
    latest_images = {}
    for file in img_files:
        match = re.search(fr'{img_name}(\d+)-(\d+){img_ext}', file)
        if match:
            img_num = int(match.group(1))
            sub_img_num = int(match.group(2))
            latest_images[img_num] = max(latest_images.get(img_num, 0), sub_img_num)
    return latest_images

def create_gif(frame_duration_ms=250, image_folder="./logs/", output_path="./results/epochs_", output_ext=".gif"):
    latest_images = find_latest_imgs(image_folder, "generated_plot_e", ".png")

    if not latest_images:
        print("No images found in the specified folder.")
        return

    for img_num, max_sub_num in latest_images.items():
        filenames = [f"generated_plot_e{str(img_num).zfill(3)}-{i}.png" for i in range(1, max_sub_num + 1)]
        images = [Image.open(os.path.join(image_folder, filename)) for filename in filenames if os.path.exists(os.path.join(image_folder, filename))]

        if images:
            gif_name = f"{output_path}{str(img_num).zfill(3)}{output_ext}"
            images[0].save(gif_name, save_all=True, append_images=images[1:], duration=frame_duration_ms, loop=1)
            print(f"GIF successfully created for epoch {img_num} and saved to {gif_name}")
        else:
            print(f"No images found for epoch {img_num} to create a gif.")

# Usage
create_gif()  # frame_duration is in milliseconds
