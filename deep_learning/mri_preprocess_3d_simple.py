import sys
import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import random
from HD_BET.hd_bet import hd_bet
import argparse
import torch

def brain_extraction(input_dir, output_dir, device):
    """
    Brain extraction using HDBET package (UNet based DL method)
    Args:
        input_dir {path} -- input directory for registered images
        output_dir {path} -- output directory for brain extracted images
    Returns:
        Brain images
    """
    print("Running brain extraction...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run HD-BET directly with the output directory
    hd_bet(input_dir, output_dir, device=device, mode='fast', tta=0)
    
    print('Brain extraction complete!')
    print("\nContents of output directory after brain extraction:")
    print(os.listdir(output_dir))

def registration(input_dir, output_dir, temp_img, interp_type='linear'):
    """
    MRI registration with SimpleITK
    Args:
        input_dir {path} -- Directory containing input images
        output_dir {path} -- Directory to save registered images
        temp_img {str} -- Registration image template
    Returns:
        The sitk image object -- nii.gz
    """
    
    # Read the template image
    fixed_img = sitk.ReadImage(temp_img, sitk.sitkFloat32)
    
    # Track problematic files
    IDs = []
    print("Preloading step...")
    for img_dir in tqdm(sorted(glob.glob(input_dir + '/*.nii.gz'))):
        ID = img_dir.split('/')[-1].split('.')[0]
        try:
            moving_img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        except Exception as e:
            IDs.append(ID)
            print(f"Error loading {ID}: {e}")
    
    count = 0
    print("Registering images...")
    list_of_files = sorted(glob.glob(input_dir + '/*.nii.gz'))
    
    for img_dir in tqdm(list_of_files):
        ID = img_dir.split('/')[-1].split('.')[0]
        if ID in IDs:
            print(f'Skipping problematic file: {ID}')
            continue
        
        if "_mask" in ID:
            continue
            
        print(f"Processing image {count + 1}: {ID}")
        
        try:
            # Read and preprocess moving image
            moving_img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            moving_img = sitk.N4BiasFieldCorrection(moving_img)

            # Resample fixed image to 1mm isotropic
            old_size = fixed_img.GetSize()
            old_spacing = fixed_img.GetSpacing()
            new_spacing = (1, 1, 1)
            new_size = [
                int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
                int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
                int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
            ]

            # Set interpolation type
            if interp_type == 'linear':
                interp_type = sitk.sitkLinear
            elif interp_type == 'bspline':
                interp_type = sitk.sitkBSpline
            elif interp_type == 'nearest_neighbor':
                interp_type = sitk.sitkNearestNeighbor

            # Resample fixed image
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(new_spacing)
            resample.SetSize(new_size)
            resample.SetOutputOrigin(fixed_img.GetOrigin())
            resample.SetOutputDirection(fixed_img.GetDirection())
            resample.SetInterpolator(interp_type)
            resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
            resample.SetOutputPixelType(sitk.sitkFloat32)
            fixed_img = resample.Execute(fixed_img)

            # Initialize transform
            transform = sitk.CenteredTransformInitializer(
                fixed_img, 
                moving_img, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY)

            # Set up registration method
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(transform)

            # Execute registration
            final_transform = registration_method.Execute(fixed_img, moving_img)

            # Apply transform and save registered image
            moving_img_resampled = sitk.Resample(
                moving_img, 
                fixed_img, 
                final_transform, 
                sitk.sitkLinear, 
                0.0, 
                moving_img.GetPixelID())
            
            # Save with _0000 suffix as required by HD-BET
            output_filename = os.path.join(output_dir, f"{ID}_0000.nii.gz")
            sitk.WriteImage(moving_img_resampled, output_filename)
            print(f"Saved registered image to: {output_filename}")
            count += 1

        except Exception as e:
            print(f"Error processing {ID}: {e}")
            continue

    print(f"Successfully registered {count} images.")
    # Debug information
    print(f"Contents of output directory {output_dir}:")
    print(os.listdir(output_dir))
    return count > 0

def main(temp_img, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"

    temp_reg_dir = os.path.join(output_dir, 'temp_registered')
    os.makedirs(temp_reg_dir, exist_ok=True)

    print("Starting brain MRI preprocessing...")

    # Step 1: Registration
    print("\nStep 1: Image Registration")
    success = registration(
        input_dir=input_dir,
        output_dir=temp_reg_dir,
        temp_img=temp_img
    )

    if not success:
        print("Registration failed! No images were processed successfully.")
        return

    print("\nChecking temporary directory contents:")
    print(os.listdir(temp_reg_dir))

    # >>> DISABLE STEP 2 <<<
    print("\nSkipping brain extraction (HD-BET). Using registered images only.")
    print("\nPreprocessing complete! Final registered images in:", temp_reg_dir)
    print("Final preprocessed files:")
    print(os.listdir(temp_reg_dir))

# def main(temp_img, input_dir, output_dir):
#     """
#     Main function to process brain MRI images
#     Args:
#         temp_img {str} -- Path to template image
#         input_dir {str} -- Path to input directory containing images
#         output_dir {str} -- Path to output directory for results
#     """
   
#     os.makedirs(output_dir, exist_ok=True)
    
#     # set device
#     device = "0" if torch.cuda.is_available() else "cpu"
    
#     # Create temporary directory for intermediate results
#     temp_reg_dir = os.path.join(output_dir, 'temp_registered')
#     os.makedirs(temp_reg_dir, exist_ok=True)
    
#     print("Starting brain MRI preprocessing...")
    
#     # REgistration
#     print("\nStep 1: Image Registration")
#     success = registration(
#         input_dir=input_dir,
#         output_dir=temp_reg_dir,
#         temp_img=temp_img
#     )
    
#     if not success:
#         print("Registration failed! No images were processed successfully.")
#         return
    
#     print("\nChecking temporary directory contents:")
#     print(os.listdir(temp_reg_dir))
    
#     # skullstripping
#     print("\nStep 2: Brain Extraction")
#     brain_extraction(
#         input_dir=temp_reg_dir,
#         output_dir=output_dir,
#         device=device
#     )
    
#     # Clean up temporary directory
#     import shutil
#     shutil.rmtree(temp_reg_dir)
    
#     print("\nPreprocessing complete! Final results saved in:", output_dir)
#     print("Final preprocessed files:")
#     print(os.listdir(output_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process brain MRI registration and skull stripping.")
    parser.add_argument("--temp_img", type=str, required=True, help="Path to the atlas template image.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed images.")

    args = parser.parse_args()
    main(temp_img=args.temp_img, input_dir=args.input_dir, output_dir=args.output_dir) 