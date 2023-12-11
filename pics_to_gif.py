from PIL import Image, ImageDraw, ImageFont
import os, glob, re, config

def find_latest_img(img_folder="./logs/", img_name="generated_plot_e", img_ext=".png"):
    full_path = img_folder + img_name + "*" + img_ext
    img_files = glob.glob(full_path)
    imgs = [int(re.search(fr'{img_name}(\d+)-\d+{img_ext}', file).group(1)) for file in img_files]
    return max(imgs) if imgs else None

def add_text_to_image(image, text, position=(10, 10), font_size=20, font_color="black"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)  # You can change the font and its size
    draw.text(position, text, font=font, fill=font_color)
    return image

def create_gif(frame_duration_ms=config.FRAME_DURATION, image_folder="./logs/", output_folder="./results/", output_name="epochs_0_to_", output_ext=".gif"):
    output_path = output_folder + output_name
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
                with Image.open(file_path) as img:
                    img_with_text = add_text_to_image(img, f"Generation {epoch}")
                    images.append(img_with_text)
                #images.append(Image.open(file_path))

    name = output_path + str(latest_img_number) + output_ext
    # Ensure there are images to create a gif
    if images:
        # Save the images as a gif
        images[0].save(name, save_all=True, append_images=images[1:], duration=frame_duration_ms, loop=1)
        print(f"GIF successfully created from images through epoch {latest_img_number} and saved to {output_path}{str(epoch)}")
    else:
        print("No images found to create a gif.")
# Usage
create_gif()  # frame_duration is in milliseconds
