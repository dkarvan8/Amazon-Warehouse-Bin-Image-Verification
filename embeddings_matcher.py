"""
embeddings_matcher.py
Handles CLIP embeddings, OCR extraction, and matching for bin verification
"""

import os
import cv2
import numpy as np
import torch
import json
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import clip
from ultralytics import YOLO
import easyocr
from rapidfuzz import fuzz
import re


class EmbeddingsMatcher:
    """Handles CLIP + OCR matching for bin images"""
    
    def __init__(self, yolo_model_path, device="cuda"):
        """
        Initialize models
        
        Args:
            yolo_model_path: Path to YOLO weights (best.pt)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print("="*70)
        print("INITIALIZING MODELS")
        print("="*70)
        
        # Load CLIP
        print("\nLoading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"✅ CLIP loaded on {self.device}")
        
        # Load YOLO
        print("\nLoading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        print("✅ YOLO loaded")
        
        # Load OCR
        print("\nLoading OCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("✅ OCR reader loaded")
        
        print("="*70)
    
    
    def _load_reference_embeddings(self, expected_asins, reference_folder, expected_products):
        """Load and generate embeddings for reference images"""
        reference_embeddings = {}
        
        for asin in expected_asins:
            ref_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = os.path.join(reference_folder, f"{asin}{ext}")
                if os.path.exists(test_path):
                    ref_path = test_path
                    break
            
            if not ref_path:
                print(f"  ⚠️  No reference for: {asin}")
                continue
            
            try:
                image = Image.open(ref_path).convert('RGB')
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
                product_info = expected_products.get(asin, {})
                reference_embeddings[asin] = {
                    'embedding': embedding.cpu().numpy().flatten(),
                    'image_path': ref_path,
                    'product_name': product_info.get('name', 'Unknown'),
                    'quantity': product_info.get('quantity', 1)
                }
                print(f"  ✅ {asin}: {product_info.get('name', 'Unknown')}")
            
            except Exception as e:
                print(f"  ❌ Error: {asin}: {e}")
        
        print(f"✅ Loaded {len(reference_embeddings)}/{len(expected_asins)} references")
        return reference_embeddings
    
    
    def _generate_detected_embeddings(self, bboxes, bin_image_rgb):
        """Generate CLIP embeddings for detected objects"""
        detected_objects = []
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cropped = bin_image_rgb[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            try:
                pil_image = Image.fromarray(cropped)
                image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
                detected_objects.append({
                    'object_id': idx,
                    'bbox': bbox,
                    'embedding': embedding.cpu().numpy().flatten()
                })
            
            except Exception as e:
                print(f"  Error on object {idx}: {e}")
        
        return detected_objects
    
    
    def _clip_matching(self, detected_objects, reference_embeddings, expected_asins):
        """Match detected objects to references using CLIP embeddings"""
        ref_asins = list(reference_embeddings.keys())
        ref_matrix = np.array([reference_embeddings[asin]['embedding'] for asin in ref_asins])
        
        # Build similarity matrix
        similarity_matrix = []
        for obj in detected_objects:
            obj_embedding = obj['embedding'].reshape(1, -1)
            similarities = cosine_similarity(obj_embedding, ref_matrix)[0]
            similarity_matrix.append(similarities)
        
        similarity_matrix = np.array(similarity_matrix)
        
        # Greedy assignment: assign each reference to best matching object
        assigned_objects = set()
        assigned_refs = {}
        
        matches = []
        
        # Sort all possible (object, ref, score) by score descending
        candidates = []
        for obj_idx, obj in enumerate(detected_objects):
            for ref_idx, asin in enumerate(ref_asins):
                score = similarity_matrix[obj_idx, ref_idx]
                candidates.append((score, obj_idx, ref_idx, asin))
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Assign greedily - each reference gets assigned to max 1 object
        for score, obj_idx, ref_idx, asin in candidates:
            obj = detected_objects[obj_idx]
            
            # Skip if object or reference already assigned
            if obj_idx in assigned_objects:
                continue
            if asin in assigned_refs:
                continue
            
            # Assign this match
            assigned_objects.add(obj_idx)
            assigned_refs[asin] = obj_idx
            
            match_status = "✅ CORRECT" if asin in expected_asins else "❌ WRONG"
            
            match_info = {
                'object_id': obj['object_id'],
                'bbox': obj['bbox'],
                'matched_asin': asin,
                'product_name': reference_embeddings[asin]['product_name'],
                'score': float(score),
                'status': match_status
            }
            
            matches.append(match_info)
            
            print(f"  Object {obj['object_id']}: {asin} ({score:.3f}) {match_status}")
        
        # Handle unmatched objects
        unmatched_objects = [obj for idx, obj in enumerate(detected_objects) 
                             if idx not in assigned_objects]
        
        if unmatched_objects:
            print(f"\n⚠️  {len(unmatched_objects)} objects could not be uniquely matched")
            for obj in unmatched_objects:
                obj_embedding = obj['embedding'].reshape(1, -1)
                similarities = cosine_similarity(obj_embedding, ref_matrix)[0]
                top_idx = np.argmax(similarities)
                top_asin = ref_asins[top_idx]
                top_score = similarities[top_idx]
                
                match_info = {
                    'object_id': obj['object_id'],
                    'bbox': obj['bbox'],
                    'matched_asin': top_asin,
                    'product_name': reference_embeddings[top_asin]['product_name'],
                    'score': float(top_score),
                    'status': '⚠️ DUPLICATE'
                }
                
                matches.append(match_info)
                
                print(f"  Object {obj['object_id']}: {top_asin} ({top_score:.3f}) ⚠️ DUPLICATE")
        
        return matches
    
    
    def _clean_text(self, text):
        """Clean and normalize text for matching"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        text = text.strip()
        return text
    
    
    def _match_ocr_to_products(self, ocr_text, expected_products, threshold=70):
        """Match OCR text to expected product names using fuzzy matching"""
        cleaned_text = self._clean_text(ocr_text)
        
        if not cleaned_text or len(cleaned_text) < 3:
            return None, 0
        
        best_score = 0
        best_asin = None
        
        for asin, info in expected_products.items():
            clean_name = self._clean_text(info['name'])
            
            # Fuzzy matching
            score = fuzz.partial_ratio(cleaned_text, clean_name)
            
            if score > best_score:
                best_score = score
                best_asin = asin
        
        if best_score >= threshold:
            return best_asin, best_score
        
        return None, best_score
    
    
    def _ocr_matching(self, bboxes, bin_image_rgb, expected_products):
        """Extract text using OCR and match to products"""
        ocr_matches = []
        
        print(f"\nRunning OCR on {len(bboxes)} detected objects...")
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cropped = bin_image_rgb[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            try:
                pil_crop = Image.fromarray(cropped)
                
                # Run OCR
                ocr_result = self.ocr_reader.readtext(np.array(pil_crop), detail=0)
                ocr_text = " ".join(ocr_result)
                
                # Match to products
                matched_asin, score = self._match_ocr_to_products(ocr_text, expected_products)
                
                ocr_matches.append({
                    'object_id': idx,
                    'bbox': bbox,
                    'ocr_text': ocr_text,
                    'matched_asin': matched_asin,
                    'ocr_score': score
                })
                
                if matched_asin:
                    print(f"  ✅ Object {idx}: '{ocr_text[:30]}...' → {matched_asin} (score: {score})")
                else:
                    print(f"  ⚠️  Object {idx}: '{ocr_text[:30]}...' → No match (score: {score})")
            
            except Exception as e:
                print(f"  ❌ Object {idx}: OCR error - {e}")
                ocr_matches.append({
                    'object_id': idx,
                    'bbox': bbox,
                    'ocr_text': '',
                    'matched_asin': None,
                    'ocr_score': 0
                })
        
        return ocr_matches
    
    
    def process_bin(self, bin_image_path, metadata_path, reference_folder):
        """
        Process single bin: YOLO detection + CLIP matching + OCR matching
        
        Args:
            bin_image_path: Path to bin image
            metadata_path: Path to metadata JSON
            reference_folder: Folder containing reference images (ASIN.jpg)
        
        Returns:
            Dict with CLIP results, OCR results, and metadata
        """
        
        bin_id = os.path.splitext(os.path.basename(bin_image_path))[0]
        
        print(f"\n{'='*70}")
        print(f"Processing Bin: {bin_id}")
        print(f"{'='*70}")
        
        # Check files exist
        if not os.path.exists(bin_image_path):
            print(f"❌ Bin image not found: {bin_image_path}")
            return None
        
        if not os.path.exists(metadata_path):
            print(f"❌ Metadata not found: {metadata_path}")
            return None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        bin_data = metadata.get('BIN_FCSKU_DATA', {})
        expected_asins = list(set([item.get('asin') for item in bin_data.values() 
                                  if item.get('asin')]))
        
        # Get product names and quantities
        expected_products = {}
        expected_items = []
        for item in bin_data.values():
            asin = item.get('asin')
            name = item.get('name', 'Unknown Product')
            quantity = item.get('quantity', 1)
            if asin:
                expected_products[asin] = {'name': name, 'quantity': quantity}
                expected_items.append({'asin': asin, 'name': name, 'quantity': quantity})
        
        print(f"\nExpected Items ({len(expected_asins)}):")
        for asin in expected_asins:
            info = expected_products.get(asin, {})
            print(f"  • {asin}: {info.get('name', 'Unknown')} (qty: {info.get('quantity', 1)})")
        
        # Step 1: YOLO Detection
        print("\n[1/5] YOLO Detection...")
        results = self.yolo_model.predict(source=bin_image_path, save=False, conf=0.25, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        print(f"✅ Detected {len(bboxes)} objects")
        
        # Step 2: Load reference embeddings
        print("\n[2/5] Loading reference embeddings...")
        reference_embeddings = self._load_reference_embeddings(expected_asins, reference_folder, expected_products)
        
        if len(reference_embeddings) == 0:
            print("❌ No references - skipping")
            return None
        
        # Step 3: Generate detected embeddings (CLIP)
        print("\n[3/5] Generating detected embeddings (CLIP)...")
        bin_image = cv2.imread(bin_image_path)
        bin_image_rgb = cv2.cvtColor(bin_image, cv2.COLOR_BGR2RGB)
        
        detected_objects = self._generate_detected_embeddings(bboxes, bin_image_rgb)
        print(f"✅ Generated {len(detected_objects)} detected embeddings")
        
        # Step 4: CLIP Matching
        print("\n[4/5] CLIP Matching...")
        clip_matches = self._clip_matching(detected_objects, reference_embeddings, expected_asins)
        
        # Step 5: OCR Matching
        print("\n[5/5] OCR Matching...")
        ocr_matches = self._ocr_matching(bboxes, bin_image_rgb, expected_products)
        
        # CLIP Validation
        expected_set = set(expected_asins)
        clip_matched_set = set([m['matched_asin'] for m in clip_matches])
        
        correct_matches = expected_set & clip_matched_set
        missing_asins = expected_set - clip_matched_set
        
        print(f"\n{'='*70}")
        print(f"CLIP MATCHING RESULTS - Bin {bin_id}")
        print(f"{'='*70}")
        print(f"\n✅ Correctly Matched ({len(correct_matches)}):")
        for asin in correct_matches:
            info = expected_products.get(asin, {})
            print(f"   • {asin}: {info.get('name', 'Unknown')}")
        
        if missing_asins:
            print(f"\n❌ Missing/Not Detected ({len(missing_asins)}):")
            for asin in missing_asins:
                info = expected_products.get(asin, {})
                print(f"   • {asin}: {info.get('name', 'Unknown')}")
        
        clip_accuracy = round(len(correct_matches) / len(expected_asins) * 100, 2) if len(expected_asins) > 0 else 0
        print(f"\n📊 CLIP Accuracy: {clip_accuracy}%")
        print(f"{'='*70}")
        
        # Return results
        return {
            'bin_id': bin_id,
            'bin_image_path': bin_image_path,
            'expected_items': expected_items,
            'expected_count': len(expected_asins),
            'detected_count': len(bboxes),
            'clip_matches': clip_matches,
            'ocr_matches': ocr_matches,
            'reference_embeddings': reference_embeddings,
            'clip_accuracy': clip_accuracy,
            'bin_image_rgb': bin_image_rgb
        }