import os
import xml.etree.ElementTree as ET

def convert_xml_to_txt(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(input_folder, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image dimensions
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            txt_lines = []
            for obj in root.findall('object'):
                # Get bounding box coordinates
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                # Convert to YOLO format: class x_center y_center width height
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Assuming "licence" is the only class and its index is 1
                class_id = 1
                txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write to corresponding txt file
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, 'w') as txt_file:
                txt_file.write("\n".join(txt_lines))

    print(f"Conversion completed. TXT files are saved in '{output_folder}'.")

# Replace with your input and output folder paths
input_folder = "C:/Users/Miguel/Desktop/CarDataset/annotations"
output_folder = "C:/Users/Miguel/Desktop/Output"

convert_xml_to_txt(input_folder, output_folder)
