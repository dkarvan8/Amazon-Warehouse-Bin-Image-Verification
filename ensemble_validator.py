"""
ensemble_validator.py
Combines CLIP and OCR results with quantity-aware filtering and validation
"""

import numpy as np
import json
import os

from embeddings_matcher import EmbeddingsMatcher

class EnsembleValidator:
    """Combines CLIP + OCR results and validates with quantity filtering"""
    
    def __init__(self):
        print("="*70)
        print("ENSEMBLE VALIDATOR INITIALIZED")
        print("="*70)
    
    
    def filter_by_quantity(self, matches, expected_items):
        """
        Keep top N matches per ASIN based on expected quantity
        
        Args:
            matches: List of ensemble matches
            expected_items: List of dicts with 'asin' and 'quantity'
        
        Returns:
            filtered_matches: Top N matches per ASIN
            detected_counts: Dict of {asin: count_detected}
        """
        # Get expected quantities
        expected_quantities = {}
        for item in expected_items:
            asin = item['asin']
            qty = item.get('quantity', 1)
            expected_quantities[asin] = qty
        
        # Group matches by ASIN
        asin_groups = {}
        for match in matches:
            asin = match['final_asin']
            if asin not in asin_groups:
                asin_groups[asin] = []
            asin_groups[asin].append(match)
        
        # Keep top N per ASIN (sorted by combined_score)
        filtered = []
        detected_counts = {}
        
        for asin, group in asin_groups.items():
            expected_qty = expected_quantities.get(asin, 1)
            
            # Sort by confidence descending, keep top N
            sorted_group = sorted(group, key=lambda x: x['combined_score'], reverse=True)
            top_n = sorted_group[:expected_qty]
            
            filtered.extend(top_n)
            detected_counts[asin] = len(top_n)
        
        return filtered, detected_counts
    
    
    def create_ensemble_results(self, matcher_result):
        """
        Combine CLIP and OCR results with weighted voting + quantity-aware filtering
        
        Args:
            matcher_result: Output from EmbeddingsMatcher.process_bin()
        
        Returns:
            Dict with ensemble matches and accuracy
        """
        
        bin_id = matcher_result['bin_id']
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE - Bin {bin_id}")
        print(f"{'='*70}")
        
        # Map results by object_id
        clip_map = {m['object_id']: m for m in matcher_result['clip_matches']}
        ocr_map = {m['object_id']: m for m in matcher_result['ocr_matches']}
        
        ensemble_matches = []
        
        # Process each detected object
        for obj_id in clip_map.keys():
            clip_match = clip_map[obj_id]
            ocr_match = ocr_map.get(obj_id, {})
            
            # Extract CLIP results
            clip_asin = clip_match['matched_asin']
            clip_score = clip_match['score']
            
            # Extract OCR results
            ocr_asin = ocr_match.get('matched_asin')
            ocr_score = ocr_match.get('ocr_score', 0)
            
            # Weighted ensemble voting
            clip_weight = 0.7  # CLIP gets 70% weight
            ocr_weight = 0.3   # OCR gets 30% weight
            
            # Decision logic
            if clip_asin == ocr_asin and ocr_asin is not None:
                # Both agree - HIGH confidence
                confidence = "HIGH"
                final_asin = clip_asin
                combined_score = clip_score * clip_weight + (ocr_score/100) * ocr_weight
                agreement = "✅ BOTH AGREE"
            
            elif ocr_asin is None or ocr_score < 50:
                # OCR failed or low confidence - trust CLIP
                confidence = "MEDIUM"
                final_asin = clip_asin
                combined_score = clip_score * clip_weight
                agreement = "⚠️ CLIP ONLY"
            
            elif clip_score > 0.6:
                # CLIP confident but OCR disagrees - trust CLIP
                confidence = "MEDIUM"
                final_asin = clip_asin
                combined_score = clip_score * clip_weight
                agreement = f"⚠️ CONFLICT (OCR says: {ocr_asin})"
            
            else:
                # Both uncertain - default to CLIP
                confidence = "LOW"
                final_asin = clip_asin
                combined_score = clip_score * clip_weight
                agreement = f"❌ UNCERTAIN"
            
            ensemble_matches.append({
                'object_id': obj_id,
                'final_asin': final_asin,
                'confidence': confidence,
                'combined_score': round(combined_score, 3),
                'clip_asin': clip_asin,
                'clip_score': round(clip_score, 3),
                'ocr_asin': ocr_asin if ocr_asin else 'None',
                'ocr_score': ocr_score,
                'agreement': agreement
            })
        
        print(f"\n--- Before Quantity Filtering: {len(ensemble_matches)} total matches ---")
        
        # Apply quantity-aware filtering
        filtered_matches, detected_counts = self.filter_by_quantity(
            ensemble_matches, 
            matcher_result['expected_items']
        )
        
        print(f"--- After Quantity Filtering: {len(filtered_matches)} kept ---\n")
        
        # Print filtered results
        for match in filtered_matches:
            obj_id = match['object_id']
            clip_asin = match['clip_asin']
            clip_score = match['clip_score']
            ocr_asin = match['ocr_asin']
            ocr_score = match['ocr_score']
            final_asin = match['final_asin']
            confidence = match['confidence']
            agreement = match['agreement']
            
            print(f"Object {obj_id} (KEPT):")
            print(f"  CLIP → {clip_asin} (score: {clip_score})")
            print(f"  OCR  → {ocr_asin} (score: {ocr_score})")
            print(f"  Final: {final_asin} | Confidence: {confidence}")
            print(f"  Status: {agreement}\n")
        
        # Validation with quantities
        expected_items_dict = {item['asin']: item.get('quantity', 1) 
                               for item in matcher_result['expected_items']}
        
        total_expected = sum(expected_items_dict.values())
        total_detected = len(filtered_matches)
        
        # Calculate per-ASIN accuracy
        correct_items = 0
        missing_items = []
        extra_items = []
        
        for asin, expected_qty in expected_items_dict.items():
            detected_qty = detected_counts.get(asin, 0)
            
            if detected_qty > 0:
                correct_items += min(detected_qty, expected_qty)
                if detected_qty < expected_qty:
                    missing_items.append({
                        'asin': asin,
                        'expected': expected_qty,
                        'detected': detected_qty,
                        'missing': expected_qty - detected_qty
                    })
            else:
                missing_items.append({
                    'asin': asin,
                    'expected': expected_qty,
                    'detected': 0,
                    'missing': expected_qty
                })
        
        # Check for extra detections
        for asin, count in detected_counts.items():
            if asin not in expected_items_dict:
                extra_items.append({'asin': asin, 'count': count})
        
        ensemble_accuracy = round(correct_items / total_expected * 100, 2) if total_expected > 0 else 0
        
        print(f"{'='*70}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Expected Items: {total_expected}")
        print(f"Detected Items: {total_detected}")
        print(f"Correct Items: {correct_items}")
        
        if missing_items:
            print(f"\n❌ Missing/Under-detected:")
            for item in missing_items:
                print(f"   • {item['asin']}: expected {item['expected']}, detected {item['detected']} (missing {item['missing']})")
        
        if extra_items:
            print(f"\n⚠️  Extra/Wrong detections:")
            for item in extra_items:
                print(f"   • {item['asin']}: {item['count']} detected (not expected)")
        
        print(f"\n📊 Ensemble Accuracy: {ensemble_accuracy}%")
        print(f"📊 CLIP-only Accuracy: {matcher_result['clip_accuracy']}%")
        print(f"{'='*70}")
        
        return {
            'bin_id': bin_id,
            'ensemble_matches': filtered_matches,
            'all_matches': ensemble_matches,
            'ensemble_accuracy': ensemble_accuracy,
            'clip_accuracy': matcher_result['clip_accuracy'],
            'total_expected': total_expected,
            'total_detected': total_detected,
            'correct_items': correct_items,
            'missing_items': missing_items,
            'extra_items': extra_items,
            'detected_counts': detected_counts
        }
    
    
    def save_results(self, ensemble_result, output_folder):
        """Save ensemble results to JSON"""
        os.makedirs(output_folder, exist_ok=True)
        
        bin_id = ensemble_result['bin_id']
        result_path = os.path.join(output_folder, f"{bin_id}_ensemble_results.json")
        
        with open(result_path, 'w') as f:
            json.dump(ensemble_result, f, indent=2)
        
        print(f"\n✅ Results saved: {result_path}")
        
        return result_path


# Example usage function
def process_single_bin(bin_image_path, metadata_path, reference_folder, yolo_model_path, output_folder="results"):
    """
    Complete pipeline: Load models -> Match -> Ensemble -> Save
    
    Args:
        bin_image_path: Path to bin image
        metadata_path: Path to metadata JSON
        reference_folder: Folder with reference images
        yolo_model_path: Path to YOLO model
        output_folder: Where to save results
    
    Returns:
        ensemble_result: Dict with all results
    """
    # Initialize
    matcher = EmbeddingsMatcher(yolo_model_path)
    validator = EnsembleValidator()
    
    # Step 1: CLIP + OCR Matching
    matcher_result = matcher.process_bin(bin_image_path, metadata_path, reference_folder)
    
    if matcher_result is None:
        print("❌ Processing failed")
        return None
    
    # Step 2: Ensemble Validation
    ensemble_result = validator.create_ensemble_results(matcher_result)
    
    # Step 3: Save Results
    validator.save_results(ensemble_result, output_folder)
    
    return ensemble_result