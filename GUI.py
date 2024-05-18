import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ml_model import compare_models, accuracy_scores  

class WaterQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Water Quality Prediction")

        self.top_label = tk.Label(root, text="Water Quality Prediction", font=("Helvetica", 16))
        self.top_label.grid(row=0, columnspan=2, padx=15, pady=15)
        
        # Create buttons 
        self.load_data_button = tk.Button(root, text="Load Data Set", command=self.load_data)
        self.load_data_button.grid(row=2, column=0, padx=5, pady=10)
        
        self.clean_data_button = tk.Button(root, text="Preprocess Data", command=self.clean_data)
        self.clean_data_button.grid(row=2, columnspan=2, padx=10, pady=10)

        self.test_train_button = tk.Button(root, text="Test Train Split", command=self.test_train_split)
        self.test_train_button.grid(row=2, column=1, padx=10, pady=10)

        self.train_button = tk.Button(root, text="Train and Evaluate", command=self.train_models)
        self.train_button.grid(row=3, column=0, padx=10, pady=10)

        self.compare_button = tk.Button(self.root, text="Compare Models", command=self.compare_models)
        self.compare_button.grid(row=3, columnspan=2, padx=10, pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_water_quality)
        self.predict_button.grid(row=3, column=1, padx=10, pady=10)

        self.refresh_button = tk.Button(root, text="Refresh", command=self.refresh)
        self.refresh_button.grid(row=4, columnspan=2, padx=10, pady=10)

        # text box to display output
        self.output_text = scrolledtext.ScrolledText(root, width=80, height=22, wrap=tk.WORD)
        self.output_text.grid(row=5, columnspan=2, padx=15, pady=15)

        self.data = None
        self.trained_model = None
        self.input_boxes = []
        self.image_label = None

    def load_data(self):
        from ml_model import data_load
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path, na_values='#NUM!')
                messagebox.showinfo("Success", "Data loaded successfully.")
                self.append_to_output("Data loaded successfully.\n")
                data_set = data_load()
                self.append_to_output(data_set)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
                self.append_to_output(f"Failed to load data:\n{str(e)}")

    def clean_data(self):
        from ml_model import generate_image_data
        if self.data is not None:
            # Generate image and output
            img, img1, output = generate_image_data()
            self.append_to_output(output)
        
            self.display_image_in_new_window(img)
            self.display_image_in_new_window(img1)
        else:
            messagebox.showwarning("Warning", "No data loaded.")
            self.append_to_output("No data loaded.")

    def test_train_split(self):
        from ml_model import load_and_split_data
        if self.data is not None:
            train_percent, test_percent = load_and_split_data()
            message = f"Percentage of data in training set: {train_percent:.2f}%\n"
            message += f"Percentage of data in testing set: {test_percent:.2f}%\n"
            self.append_to_output(message)
        else:
            messagebox.showwarning("Warning", "No data loaded.")
            self.append_to_output("No data loaded.")

    def train_models(self):
        from ml_model import train_evaluate_models
        if self.data is not None:
            try:
                output = train_evaluate_models(self.data)
                self.append_to_output(output)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train and evaluate models:\n{str(e)}")
                self.append_to_output(f"Failed to train and evaluate models:\n{str(e)}")
        else:
            messagebox.showwarning("Warning", "No data loaded.")
            self.append_to_output("No data loaded.")

    def compare_models(self):
        image_buffer = compare_models(accuracy_scores)

        new_window = tk.Toplevel(self.root)
        new_window.title("Model Comparison")

        # image buffer to a Tkinter Image object
        image = Image.open(image_buffer)
        photo_image = ImageTk.PhotoImage(image)

        image_label = tk.Label(new_window, image=photo_image)
        image_label.image = photo_image 
        image_label.pack()


    def predict_water_quality(self):
        from ml_model import predict_random_data, predict_random_data_xgb, train_xgb_model
        # Load data from CSV file
        df = pd.read_csv('waterQuality1-checkpoint.csv')
        X_val_reshaped = df.values
        predictions_df = predict_random_data(X_val_reshaped)

        message = f"\nPredictions using LSTM Model:\n"
        self.append_to_output(message) 
        self.append_to_output(predictions_df)   
        missing_value = ['#NUM!', np.nan]
        df = pd.read_csv('waterQuality1-checkpoint.csv', na_values=missing_value)

        X = df.drop(columns=['is_safe']) 
        y = df['is_safe'] 

        # Handle missing values in features
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        X_train = df.drop(columns=['is_safe'])  
        y_train = df['is_safe']  
        trained_model = train_xgb_model(X_train, y_train)

        predictions_df1 = predict_random_data_xgb(X_train, trained_model)

        message = f"\nPredictions using XGBoost Model:\n"
        self.append_to_output(message)
        self.append_to_output(predictions_df1)


    def refresh(self):
        self.output_text.delete(1.0, tk.END)

        messagebox.showinfo("Refresh", "Load the data set")
        for image_label in self.displayed_images:
            image_label.destroy()
        self.displayed_images = []

    def append_to_output(self, text):
        self.output_text.insert(tk.END, text)

    def display_image(self, image_buffer):
        image = Image.open(image_buffer)
        photo_image = ImageTk.PhotoImage(image)

        if self.image_label:
            self.image_label.destroy() 
        self.image_label = tk.Label(self.root, image=photo_image)
        self.image_label.image = photo_image
        self.image_label.grid(row=6, columnspan=2, padx=15, pady=15)

    def display_image_in_new_window(self, image):
        new_window = tk.Toplevel(self.root)
        new_window.title("Image")
        new_window.geometry("700x700")

        image_tk = ImageTk.PhotoImage(image)
    
        image_label = tk.Label(new_window, image=image_tk)
        image_label.image = image_tk
        image_label.pack()


def main():
    root = tk.Tk()
    root.geometry("700x600") 
    app = WaterQualityApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
