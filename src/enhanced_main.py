import os
import sys
import tkinter as tk
from tkinter import messagebox

# Check if running as script or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the GUI application
    from gui_app import FacialRecognitionGUI
    
    # Start the application
    def main():
        root = tk.Tk()
        app = FacialRecognitionGUI(root)
        root.mainloop()
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    # Show error message if something goes wrong
    messagebox.showerror("Error", f"An error occurred while starting the application:\n{str(e)}")
    
    # Print detailed error information
    import traceback
    traceback.print_exc()