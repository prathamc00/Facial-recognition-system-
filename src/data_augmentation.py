import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random


class DataAugmentation:
    def __init__(self):
        """Initialize the data augmentation utility."""
        
        # Create Keras ImageDataGenerator for advanced augmentation
        self.image_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def augment_brightness(self, image, brightness_range=(0.7, 1.3)):
        """
        Adjust image brightness randomly within a range.
        
        Args:
            image (numpy.ndarray): Input image
            brightness_range (tuple): (min, max) brightness multiplier
        
        Returns:
            numpy.ndarray: Brightness-adjusted image
        """
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust V channel (brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # Convert back to BGR
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return augmented
    
    def augment_contrast(self, image, contrast_range=(0.8, 1.2)):
        """
        Adjust image contrast randomly within a range.
        
        Args:
            image (numpy.ndarray): Input image
            contrast_range (tuple): (min, max) contrast multiplier
        
        Returns:
            numpy.ndarray: Contrast-adjusted image
        """
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Adjust L channel (lightness/contrast)
        lab[:, :, 0] = lab[:, :, 0] * contrast_factor
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        # Convert back to BGR
        augmented = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return augmented
    
    def add_gaussian_noise(self, image, noise_std=10):
        """
        Add Gaussian noise to the image.
        
        Args:
            image (numpy.ndarray): Input image
            noise_std (float): Standard deviation of Gaussian noise
        
        Returns:
            numpy.ndarray: Image with added noise
        """
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def rotate_image(self, image, angle_range=(-15, 15)):
        """
        Rotate image by a random angle.
        
        Args:
            image (numpy.ndarray): Input image
            angle_range (tuple): (min, max) rotation angle in degrees
        
        Returns:
            numpy.ndarray: Rotated image
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def flip_horizontal(self, image):
        """
        Flip image horizontally.
        
        Args:
            image (numpy.ndarray): Input image
        
        Returns:
            numpy.ndarray: Flipped image
        """
        return cv2.flip(image, 1)
    
    def apply_blur(self, image, blur_type='gaussian', kernel_size=5):
        """
        Apply blur to the image.
        
        Args:
            image (numpy.ndarray): Input image
            blur_type (str): Type of blur ('gaussian', 'median', 'bilateral')
            kernel_size (int): Kernel size for blurring
        
        Returns:
            numpy.ndarray: Blurred image
        """
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif blur_type == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            return image
    
    def augment_batch(self, images, augmentations_per_image=3):
        """
        Apply random augmentations to a batch of images.
        
        Args:
            images (list): List of input images
            augmentations_per_image (int): Number of augmented versions per image
        
        Returns:
            list: List of augmented images (including originals)
        """
        augmented_images = []
        
        # Available augmentation functions
        augmentation_funcs = [
            self.augment_brightness,
            self.augment_contrast,
            lambda img: self.add_gaussian_noise(img, noise_std=5),
            self.rotate_image,
            self.flip_horizontal,
            lambda img: self.apply_blur(img, blur_type='gaussian', kernel_size=3)
        ]
        
        for image in images:
            # Add original image
            augmented_images.append(image)
            
            # Generate augmented versions
            for _ in range(augmentations_per_image):
                aug_image = image.copy()
                
                # Apply 1-3 random augmentations
                num_augmentations = random.randint(1, 3)
                selected_augmentations = random.sample(augmentation_funcs, num_augmentations)
                
                for aug_func in selected_augmentations:
                    try:
                        aug_image = aug_func(aug_image)
                    except Exception as e:
                        print(f"Error applying augmentation: {e}")
                        continue
                
                augmented_images.append(aug_image)
        
        return augmented_images
    
    def augment_single(self, image, num_augmentations=2):
        """
        Apply multiple random augmentations to a single image.
        
        Args:
            image (numpy.ndarray): Input image
            num_augmentations (int): Number of augmentations to apply
        
        Returns:
            numpy.ndarray: Augmented image
        """
        augmentation_funcs = [
            lambda img: self.augment_brightness(img, brightness_range=(0.7, 1.3)),
            lambda img: self.augment_contrast(img, contrast_range=(0.8, 1.2)),
            lambda img: self.add_gaussian_noise(img, noise_std=8),
            lambda img: self.rotate_image(img, angle_range=(-15, 15)),
        ]
        
        aug_image = image.copy()
        
        # Randomly select and apply augmentations
        selected_augmentations = random.sample(
            augmentation_funcs,
            min(num_augmentations, len(augmentation_funcs))
        )
        
        for aug_func in selected_augmentations:
            try:
                aug_image = aug_func(aug_image)
            except Exception as e:
                print(f"Error applying augmentation: {e}")
        
        # Randomly decide whether to flip
        if random.random() > 0.5:
            aug_image = self.flip_horizontal(aug_image)
        
        return aug_image
    
    def create_training_set(self, images, labels, augmentation_factor=5):
        """
        Create an augmented training set from original images.
        
        Args:
            images (list): List of original images
            labels (list): List of corresponding labels
            augmentation_factor (int): Number of augmented versions per image
        
        Returns:
            tuple: (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Add original
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                aug_image = self.augment_single(image, num_augmentations=2)
                augmented_images.append(aug_image)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels


# Example usage
if __name__ == "__main__":
    # Create augmentation utility
    aug = DataAugmentation()
    
    # Load a sample image
    image_path = "../data/test_image.jpg"
    if cv2.os.path.exists(image_path):
        image = cv2.imread(image_path)
        
        # Generate augmented versions
        augmented = aug.augment_batch([image], augmentations_per_image=3)
        
        print(f"Generated {len(augmented)} images from 1 original")
        
        # Display results
        for i, aug_img in enumerate(augmented[:5]):  # Show first 5
            cv2.imshow(f"Augmented {i}", aug_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Sample image not found. Create test images to see augmentation in action.")
