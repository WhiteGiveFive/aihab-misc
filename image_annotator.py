import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import argparse

# Dropdown labels dictionary
Dropdown_labels = {
    'No label assigned': None,
    'Urban': 0,
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Sea': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Littoral Rock': 13,
    'Littoral Sediment': 14,
    'Montane': 15,
    'Standing Open Waters and Canals': 16,
    'Inland Rock': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19
}


# Create ImageViewer class
class ImageViewer:
    def __init__(self, root, folder_path):
        self.root = root
        self.folder_path = folder_path
        self.all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images = self.all_images.copy()
        self.current_index = 0
        self.data_file = os.path.join(folder_path, "image_annotations.csv")
        self.data = self.load_existing_data()

        # GUI Components
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack()

        self.label_dropdown = ttk.Combobox(root, values=list(Dropdown_labels.keys()), state="readonly")
        self.label_dropdown.set("no label assigned")
        self.label_dropdown.pack()

        self.comment_entry = tk.Entry(root, width=50)
        self.comment_entry.pack()

        # Navigation, Save, and Filter Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.prev_button = tk.Button(button_frame, text="Previous", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="Save", command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(button_frame, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Checkbox to filter unlabelled images
        self.show_unlabelled_var = tk.BooleanVar()
        self.filter_checkbox = tk.Checkbutton(root, text="Show Only Unlabelled", variable=self.show_unlabelled_var,
                                              command=self.filter_images)
        self.filter_checkbox.pack()

        # Key bindings
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.previous_image)

        self.update_image()
        self.update_info()

    def load_existing_data(self):
        """Load existing data from the CSV file, if available."""
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        else:
            return pd.DataFrame(columns=["File Name", "Label", "Comment"])

    def filter_images(self):
        """Filter images based on whether they are labelled or not."""
        if self.show_unlabelled_var.get():
            # Filter for unlabelled images
            labelled_files = set(self.data[self.data["Label"].notna()]["File Name"])
            self.images = [img for img in self.all_images if img not in labelled_files]
        else:
            # Show all images
            self.images = self.all_images.copy()

        # Reset the index and update the display
        self.current_index = 0
        self.update_image()
        self.update_info()

    def update_info(self):
        """Update the information label with total and labeled images count."""
        total_images = len(self.all_images)
        labelled_images = self.data["Label"].notna().sum()
        displayed_images = len(self.images)
        self.info_label.config(
            text=f"Total Images: {total_images} | Labelled: {labelled_images} | Displaying: {displayed_images}"
        )

    def update_image(self):
        """Update the displayed image."""
        if not self.images:
            self.image_label.config(text="No images to display.")
            self.label_dropdown.set("no label assigned")
            self.comment_entry.delete(0, tk.END)
            return

        img_path = os.path.join(self.folder_path, self.images[self.current_index])
        img = Image.open(img_path)
        img = img.resize((384, 384))
        self.photo = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
        self.root.title(f"Viewing: {self.images[self.current_index]}")

        # Load existing data for this image
        filename = self.images[self.current_index]
        row = self.data[self.data["File Name"] == filename]
        if not row.empty:
            self.label_dropdown.set(row.iloc[0]["Label"] if row.iloc[0]["Label"] else "no label assigned")
            self.comment_entry.delete(0, tk.END)
            self.comment_entry.insert(0, row.iloc[0]["Comment"])
        else:
            self.label_dropdown.set("no label assigned")
            self.comment_entry.delete(0, tk.END)

    def next_image(self, event=None):
        """Navigate to the next image."""
        if not self.images:
            return
        self.current_index = (self.current_index + 1) % len(self.images)
        self.update_image()

    def previous_image(self, event=None):
        """Navigate to the previous image."""
        if not self.images:
            return
        self.current_index = (self.current_index - 1) % len(self.images)
        self.update_image()

    def save_data(self):
        """Save the current label and comment to the DataFrame and update the CSV."""
        if not self.images:
            return

        filename = self.images[self.current_index]
        label = self.label_dropdown.get()
        comment = self.comment_entry.get()

        # Update or add data for the current image
        self.data = self.data[self.data["File Name"] != filename]
        self.data = pd.concat([self.data, pd.DataFrame([{
            "File Name": filename,
            "Label": label if label != "no label assigned" else "",
            "Comment": comment
        }])], ignore_index=True)

        # Save to CSV
        self.data.to_csv(self.data_file, index=False)
        messagebox.showinfo("Save", f"Data for {filename} saved!")
        self.update_info()


# Main function to start the application
def main():
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    if not folder_path:
        return

    root = tk.Tk()
    app = ImageViewer(root, folder_path)
    root.mainloop()


if __name__ == "__main__":
    main()