from PIL import Image
import os, glob, re, config

def find_latest_img(img_folder="./logs/", img_name="generated_plot_e", img_ext=".png"):
    full_path = img_folder + img_name + "*" + img_ext
    img_files = glob.glob(full_path)
    imgs = [int(re.search(fr'{img_name}(\d+)-\d+{img_ext}', file).group(1)) for file in img_files]
    return max(imgs) if imgs else None

def create_gif(frame_duration_ms=250, image_folder="./logs/", output_path="./results/epochs_0_to_", output_ext=".gif"):
    latest_img_number = find_latest_img(image_folder)
    # Check
    if latest_img_number is None:
        print("No images found in the specified folder.")
        return
    
    # Load images
    images = []
    for epoch in range(1, latest_img_number + 1):
        for version in range(config.INTERPOLATION_STEPS):
            filename = f"generated_plot_e{str(epoch)}-{version}.png"
            file_path = os.path.join(image_folder, filename)
            if os.path.exists(file_path):
                images.append(Image.open(file_path))

    name = output_path + str(latest_img_number) + output_ext
    # Ensure there are images to create a gif
    if images:
        # Save the images as a gif
        images[0].save(name, save_all=True, append_images=images[1:], duration=frame_duration_ms, loop=1)
        print(f"GIF successfully created from images through epoch {latest_img_number} and saved to {output_path}")
    else:
        print("No images found to create a gif.")
# Usage
create_gif()  # frame_duration is in milliseconds
