"""
Liver CT portal-venous to arterial phase registration using SimpleITK-SimpleElastix.
Arterial phase (arterial-phase.nii.gz) = fixed image.
Portal-venous phase (portal-venous.nii.gz) = moving image.
Output: registered portal-venous image (portal-venous_reg.nii.gz).
"""

import argparse
import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk


class HCCCTRegistration:
    def __init__(self, data_path: str, num_workers: int = 4):
        self.data_path = Path(data_path)
        self.num_workers = num_workers

    def preprocess_images(self, fixed_path: Path, moving_path: Path) -> Tuple:
        """
        Preprocess CT images for registration.
        Returns windowed images for registration and original images for final output.
        """
        try:
            print("  Reading original images...")
            fixed_sitk_original = sitk.ReadImage(str(fixed_path))
            moving_sitk_original = sitk.ReadImage(str(moving_path))

            print(f"  Fixed image size: {fixed_sitk_original.GetSize()}, "
                  f"spacing: {fixed_sitk_original.GetSpacing()}")
            print(f"  Moving image size: {moving_sitk_original.GetSize()}, "
                  f"spacing: {moving_sitk_original.GetSpacing()}")

            # Resample moving image to fixed image space
            print("  Resampling moving image...")
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk_original)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1024)  # CT air value

            moving_resampled = resampler.Execute(moving_sitk_original)

            def apply_liver_window(image_sitk, width=180, level=40):
                """
                Apply liver window/level to a CT image.
                Args:
                    image_sitk: SimpleITK image
                    width: window width (HU)
                    level: window level (HU)
                Returns:
                    Windowed SimpleITK image
                """
                image_array = sitk.GetArrayFromImage(image_sitk)
                lower_bound = level - width / 2
                upper_bound = level + width / 2
                windowed_array = np.clip(image_array, lower_bound, upper_bound)
                windowed_image = sitk.GetImageFromArray(windowed_array.astype(np.float32))
                windowed_image.CopyInformation(image_sitk)
                return windowed_image

            print("  Applying liver window (width=150HU, level=50HU)...")
            fixed_windowed = apply_liver_window(fixed_sitk_original, width=150, level=50)
            moving_windowed = apply_liver_window(moving_resampled, width=150, level=50)

            return fixed_windowed, moving_windowed, moving_resampled, fixed_sitk_original

        except Exception as e:
            print(f"  Preprocessing failed: {e}")
            raise

    def create_registration_parameters(self) -> List[sitk.ParameterMap]:
        """
        Create optimized multi-stage registration parameters for liver CT.
        Three stages: Rigid -> Affine -> B-spline.
        """
        parameter_maps = []

        # Stage 1: Rigid
        rigid_map = sitk.GetDefaultParameterMap("rigid")
        rigid_map["AutomaticParameterEstimation"] = ["true"]
        rigid_map["AutomaticTransformInitialization"] = ["true"]
        rigid_map["CheckNumberOfSamples"] = ["true"]
        rigid_map["DefaultPixelValue"] = ["-1024"]
        rigid_map["FinalBSplineInterpolationOrder"] = ["3"]
        rigid_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
        rigid_map["ImageSampler"] = ["RandomCoordinate"]
        rigid_map["Interpolator"] = ["BSplineInterpolator"]
        rigid_map["MaximumNumberOfIterations"] = ["256"]
        rigid_map["Metric"] = ["AdvancedMattesMutualInformation"]
        rigid_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
        rigid_map["NumberOfHistogramBins"] = ["32"]
        rigid_map["NumberOfResolutions"] = ["4"]
        rigid_map["NumberOfSamplesForExactGradient"] = ["4096"]
        rigid_map["NumberOfSpatialSamples"] = ["2048"]
        rigid_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        rigid_map["Registration"] = ["MultiResolutionRegistration"]
        rigid_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        rigid_map["Resampler"] = ["DefaultResampler"]
        rigid_map["Transform"] = ["EulerTransform"]
        rigid_map["WriteResultImage"] = ["false"]
        parameter_maps.append(rigid_map)

        # Stage 2: Affine
        affine_map = sitk.GetDefaultParameterMap("affine")
        affine_map["AutomaticParameterEstimation"] = ["true"]
        affine_map["AutomaticTransformInitialization"] = ["true"]
        affine_map["CheckNumberOfSamples"] = ["true"]
        affine_map["DefaultPixelValue"] = ["-1024"]
        affine_map["FinalBSplineInterpolationOrder"] = ["3"]
        affine_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
        affine_map["ImageSampler"] = ["RandomCoordinate"]
        affine_map["Interpolator"] = ["BSplineInterpolator"]
        affine_map["MaximumNumberOfIterations"] = ["256"]
        affine_map["Metric"] = ["AdvancedMattesMutualInformation"]
        affine_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
        affine_map["NumberOfHistogramBins"] = ["32"]
        affine_map["NumberOfResolutions"] = ["4"]
        affine_map["NumberOfSamplesForExactGradient"] = ["4096"]
        affine_map["NumberOfSpatialSamples"] = ["2048"]
        affine_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        affine_map["Registration"] = ["MultiResolutionRegistration"]
        affine_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        affine_map["Resampler"] = ["DefaultResampler"]
        affine_map["Transform"] = ["AffineTransform"]
        affine_map["WriteResultImage"] = ["false"]
        parameter_maps.append(affine_map)

        # Stage 3: B-spline
        bspline_map = sitk.GetDefaultParameterMap("bspline")
        bspline_map["AutomaticParameterEstimation"] = ["true"]
        bspline_map["DefaultPixelValue"] = ["-1024"]
        bspline_map["FinalBSplineInterpolationOrder"] = ["3"]
        bspline_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
        bspline_map["GridSpacingSchedule"] = ["4.0", "2.0", "1.0", "0.5"]
        bspline_map["ImageSampler"] = ["RandomCoordinate"]
        bspline_map["Interpolator"] = ["BSplineInterpolator"]
        bspline_map["MaximumNumberOfIterations"] = ["128"]
        bspline_map["Metric"] = ["AdvancedMattesMutualInformation"]
        bspline_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
        bspline_map["NumberOfHistogramBins"] = ["32"]
        bspline_map["NumberOfResolutions"] = ["4"]
        bspline_map["NumberOfSamplesForExactGradient"] = ["4096"]
        bspline_map["NumberOfSpatialSamples"] = ["2048"]
        bspline_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        bspline_map["Registration"] = ["MultiResolutionRegistration"]
        bspline_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        bspline_map["Resampler"] = ["DefaultResampler"]
        bspline_map["Transform"] = ["BSplineTransform"]
        bspline_map["WriteResultImage"] = ["true"]
        bspline_map["FinalGridSpacingInPhysicalUnits"] = ["20.0"]
        bspline_map["UseDirectionCosines"] = ["true"]
        bspline_map["BSplineTransformSplineOrder"] = ["3"]
        bspline_map["UseCyclicTransform"] = ["false"]
        parameter_maps.append(bspline_map)

        return parameter_maps

    def calculate_mutual_information(self, image1_array: np.ndarray,
                                     image2_array: np.ndarray,
                                     num_bins: int = 64) -> float:
        """
        Compute mutual information between two 3D images.
        Focuses on liver-relevant HU range (-100 to 200 HU).
        """
        if image1_array.shape != image2_array.shape:
            from scipy.ndimage import zoom
            zoom_factors = [image2_array.shape[i] / image1_array.shape[i]
                            for i in range(3)]
            image2_array = zoom(image2_array, zoom_factors, order=1)

        x = image1_array.flatten()
        y = image2_array.flatten()

        # Focus on liver tissue HU range
        mask = (x > -100) & (x < 200) & (y > -100) & (y < 200)
        x_filtered = x[mask]
        y_filtered = y[mask]

        if len(x_filtered) == 0:
            x_filtered = x
            y_filtered = y

        hist_2d, _, _ = np.histogram2d(x_filtered, y_filtered, bins=num_bins)
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        mi = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return float(mi)

    def evaluate_registration(self, fixed_image: sitk.Image,
                              moving_image: sitk.Image,
                              registered_image: sitk.Image) -> Dict:
        """
        Evaluate registration quality via mutual information before and after.
        """
        try:
            fixed_array = sitk.GetArrayFromImage(fixed_image)
            moving_array = sitk.GetArrayFromImage(moving_image)
            registered_array = sitk.GetArrayFromImage(registered_image)

            mi_before = self.calculate_mutual_information(fixed_array, moving_array)
            mi_after = self.calculate_mutual_information(fixed_array, registered_array)
            mi_improvement = ((mi_after - mi_before) / mi_before * 100) if mi_before > 0 else 0

            return {
                'MI_before': float(mi_before),
                'MI_after': float(mi_after),
                'MI_improvement_pct': float(mi_improvement)
            }
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            return {'MI_before': 0.0, 'MI_after': 0.0, 'MI_improvement_pct': 0.0}

    def register_patient(self, patient_folder: Path) -> Dict:
        """
        Register a single patient's portal-venous phase to arterial phase.
        """
        patient_id = patient_folder.name
        result = {
            'patient_id': patient_id,
            'status': 'unknown',
            'time_sec': 0,
            'MI_before': None,
            'MI_after': None,
            'MI_improvement_pct': None,
            'error': None
        }

        try:
            print(f"\nProcessing patient: {patient_id}")
            start_time = time.time()

            arterial_path = patient_folder / "arterial-phase.nii.gz"
            portal_path = patient_folder / "portal-venous.nii.gz"

            if not arterial_path.exists():
                result['status'] = 'failed'
                result['error'] = f"Arterial phase image not found: {arterial_path}"
                print(f"  {result['error']}")
                return result

            if not portal_path.exists():
                result['status'] = 'failed'
                result['error'] = f"Portal-venous image not found: {portal_path}"
                print(f"  {result['error']}")
                return result

            # Preprocess: windowed images for registration, originals for output
            fixed_windowed, moving_windowed, moving_original, fixed_original = \
                self.preprocess_images(arterial_path, portal_path)

            # Multi-stage registration
            print("  Configuring registration parameters...")
            parameter_maps = self.create_registration_parameters()

            print("  Running registration...")
            elastix_image_filter = sitk.ElastixImageFilter()
            elastix_image_filter.SetFixedImage(fixed_windowed)
            elastix_image_filter.SetMovingImage(moving_windowed)

            for param_map in parameter_maps:
                elastix_image_filter.AddParameterMap(param_map)

            elastix_image_filter.LogToConsoleOff()
            elastix_image_filter.Execute()

            registered_windowed = elastix_image_filter.GetResultImage()
            transform_parameter_maps = elastix_image_filter.GetTransformParameterMap()

            # Evaluate
            print("  Computing mutual information...")
            metrics = self.evaluate_registration(
                fixed_windowed, moving_windowed, registered_windowed)

            # Apply the same transforms to the original image
            print("  Applying transforms to original image...")
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetTransformParameterMap(transform_parameter_maps)
            transformix_image_filter.SetMovingImage(moving_original)
            transformix_image_filter.LogToConsoleOff()
            transformix_image_filter.Execute()

            registered_original = transformix_image_filter.GetResultImage()

            # Save registered image (original CT values)
            output_path = patient_folder / "portal-venous_reg.nii.gz"
            sitk.WriteImage(registered_original, str(output_path))
            print(f"  Saved registered image: {output_path}")

            result['status'] = 'success'
            result['time_sec'] = round(time.time() - start_time, 2)
            result.update(metrics)

            print(f"  Registration successful! Time: {result['time_sec']}s")
            print(f"  MI: before={result['MI_before']:.4f}, "
                  f"after={result['MI_after']:.4f}, "
                  f"improvement={result['MI_improvement_pct']:.2f}%")

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"  Registration failed: {e}")
            traceback.print_exc()

        return result

    def process_all_patients(self) -> pd.DataFrame:
        """
        Batch process all patients, optionally in parallel.
        """
        print(f"Processing directory: {self.data_path}")

        patient_folders = sorted([
            folder for folder in self.data_path.iterdir()
            if folder.is_dir()
        ])

        if not patient_folders:
            print(f"Error: No folders found in {self.data_path}")
            return pd.DataFrame()

        print(f"Found {len(patient_folders)} patient folders")

        if self.num_workers > 1 and len(patient_folders) > 1:
            print(f"Using {self.num_workers} processes in parallel...")
            with mp.Pool(min(self.num_workers, len(patient_folders))) as pool:
                results = pool.map(self.register_patient, patient_folders)
        else:
            print("Using single process...")
            results = [self.register_patient(folder) for folder in patient_folders]

        df = pd.DataFrame(results)
        self._save_and_summarize(df)
        return df

    def _save_and_summarize(self, df: pd.DataFrame):
        """Save results to Excel and print summary statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = Path(f"registration_results_{timestamp}.xlsx")

        column_order = ['patient_id', 'status', 'time_sec',
                        'MI_before', 'MI_after', 'MI_improvement_pct', 'error']
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        df.to_excel(excel_path, index=False)
        print(f"\nResults saved to: {excel_path.absolute()}")

        # Summary
        print("\n" + "=" * 60)
        print("Registration Summary")
        print("=" * 60)

        total = len(df)
        n_success = len(df[df['status'] == 'success'])
        n_failed = len(df[df['status'] == 'failed'])

        print(f"Total patients: {total}")
        print(f"Successful: {n_success} ({n_success / total * 100:.1f}%)")
        print(f"Failed: {n_failed} ({n_failed / total * 100:.1f}%)")

        if n_success > 0:
            s = df[df['status'] == 'success']
            print(f"\nMean metrics (successful cases):")
            print(f"  Avg time: {s['time_sec'].mean():.2f} s")
            print(f"  Avg MI before: {s['MI_before'].mean():.4f}")
            print(f"  Avg MI after: {s['MI_after'].mean():.4f}")
            print(f"  Avg MI improvement: {s['MI_improvement_pct'].mean():.2f}%")

            if 'MI_improvement_pct' in s.columns:
                best_idx = s['MI_improvement_pct'].idxmax()
                worst_idx = s['MI_improvement_pct'].idxmin()
                print(f"\nBest: {s.loc[best_idx, 'patient_id']} "
                      f"(+{s.loc[best_idx, 'MI_improvement_pct']:.2f}%)")
                print(f"Worst: {s.loc[worst_idx, 'patient_id']} "
                      f"(+{s.loc[worst_idx, 'MI_improvement_pct']:.2f}%)")

        if n_failed > 0:
            print(f"\nFailed patients:")
            for _, row in df[df['status'] == 'failed'].iterrows():
                print(f"  {row['patient_id']}: {row.get('error', 'unknown')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Liver CT portal-venous to arterial phase registration "
                    "using SimpleITK-SimpleElastix")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory containing patient folders "
                             "(each with arterial-phase.nii.gz and portal-venous.nii.gz)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel processes (default: 4, set 1 for single)")
    parser.add_argument("--output_excel", type=str, default=None,
                        help="Output Excel path for registration results "
                             "(default: registration_results_<timestamp>.xlsx)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return

    try:
        import SimpleITK
    except ImportError:
        print("Error: SimpleITK not installed.")
        print("Install with: pip install SimpleITK-SimpleElastix")
        return

    print("=" * 60)
    print("Liver CT Portal-Venous to Arterial Phase Registration")
    print("Using SimpleITK-SimpleElastix")
    print("=" * 60)
    print(f"Data directory: {args.data_path}")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)
    print("Strategy:")
    print("  1. Windowed images for registration computation")
    print("  2. Apply deformation field to original images")
    print("  3. Save registered output with original CT values")
    print("  4. Liver window: width=150HU, level=50HU")
    print("  5. Mutual information evaluation before/after")
    print("  6. Three-stage: Rigid -> Affine -> B-spline")
    print("=" * 60)

    registrar = HCCCTRegistration(args.data_path, num_workers=args.num_workers)

    start_time = time.time()
    results_df = registrar.process_all_patients()
    total_time = time.time() - start_time

    print(f"\nTotal processing time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("Registration complete!")

    if not results_df.empty:
        print(f"\nOutput summary:")
        print(f"  - Excel report: registration_results_*.xlsx")
        print(f"  - Registered images: portal-venous_reg.nii.gz in each patient folder")


if __name__ == "__main__":
    main()
