"""
app.py
Streamlit application for Amazon Bin Verification System
Two sections: Showcase Results + On-the-Fly Processing
"""

import streamlit as st
import os
import json
from PIL import Image
import cv2
import numpy as np
from scraper import scrape_reference_images
from embeddings_matcher import EmbeddingsMatcher
from ensemble_validator import EnsembleValidator
import shutil
import clip

# Page configuration
st.set_page_config(
    page_title="Amazon Bin Verification",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #232F3E;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #FF9900;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF9900;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize models (cached to load only once)
@st.cache_resource
def load_models():
    """Load models once and cache them"""
    try:
        matcher = EmbeddingsMatcher(yolo_model_path="best.pt")
        validator = EnsembleValidator()
        return matcher, validator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


# Main header
st.markdown('<p class="main-header">📦 Amazon Bin Verification System</p>', unsafe_allow_html=True)


# Sidebar navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=200)
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "**Select Mode:**",
    ["🎯 Showcase Results", "⚡ Process New Bin"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Sample Results:** View pre-processed verification results for 9 sample bins.\n\n"
    "**Process New Bin:** Upload a new bin image and metadata to run real-time verification."
)


# ============================================================================
# SECTION 1: SHOWCASE RESULTS
# ============================================================================

if section == "🎯 Showcase Results":
    
    st.markdown('<p class="section-header">Sample Verification Results</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This section displays results from 9 pre-processed bins demonstrating the system's capabilities.
    Select a bin to view detailed matching results.
    """)
    
    # Get available bins
    display_folder = "Display"
    
    if not os.path.exists(display_folder):
        st.error(f"❌ Display folder not found: {display_folder}")
        st.stop()
    
    bins = sorted([d for d in os.listdir(display_folder) 
                   if os.path.isdir(os.path.join(display_folder, d)) and d.startswith("bin_")])
    
    if len(bins) == 0:
        st.warning("⚠️ No showcase bins found in Display folder")
        st.stop()
    
    # Bin selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_bin = st.selectbox(
            "**Select Bin to View:**",
            bins,
            format_func=lambda x: f"Bin {x.replace('bin_', '')}"
        )
    
    if selected_bin:
        
        bin_folder = os.path.join(display_folder, selected_bin)
        results_path = os.path.join(bin_folder, "results.json")
        viz_path = os.path.join(bin_folder, "visualization.jpg")
        
        # Load results
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            bin_id = results['bin_id']
            
            # Display metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Expected Items", results['expected_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Detected Objects", results['detected_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Correct Matches", len(results['correctly_matched']))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                accuracy = results['accuracy']
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Two columns: Results + Visualization
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("### 📋 Expected Items")
                for item in results['expected_items']:
                    st.markdown(f"- **{item['asin']}**: {item['name']}")
                
                st.markdown("---")
                
                # Correctly matched
                if len(results['correctly_matched']) > 0:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ Correctly Matched")
                    for item in results['correctly_matched']:
                        st.markdown(f"- **{item['asin']}**: {item['name']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Missing items
                if len(results['missing_items']) > 0:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown("### ❌ Missing/Not Detected")
                    for item in results['missing_items']:
                        st.markdown(f"- **{item['asin']}**: {item['name']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Extra detections
                if len(results.get('extra_detections', [])) > 0:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ Extra Detections")
                    for asin in results['extra_detections']:
                        st.markdown(f"- {asin}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 🖼️ Visualization")
                if os.path.exists(viz_path):
                    viz_image = Image.open(viz_path)
                    st.image(viz_image, use_container_width=True)
                else:
                    st.warning("Visualization image not found")
            
            st.markdown("---")
            
            # Match details table
            st.markdown("### 📊 Detailed Match Results")
            
            if 'match_details' in results and len(results['match_details']) > 0:
                
                for idx, match in enumerate(results['match_details']):
                    with st.expander(f"Object {match['object_id']}: {match['matched_asin']} - Score: {match['score']:.3f}"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**ASIN:** {match['matched_asin']}")
                            st.markdown(f"**Product:** {match['product_name'][:50]}...")
                            st.markdown(f"**Confidence:** {match['score']:.3f}")
                            st.markdown(f"**Status:** {match.get('status', 'N/A')}")
                        
                        with col2:
                            # Show reference image if available
                            ref_folder = "reference_images"
                            for ext in ['.jpg', '.jpeg', '.png']:
                                ref_path = os.path.join(ref_folder, f"{match['matched_asin']}{ext}")
                                if os.path.exists(ref_path):
                                    st.image(ref_path, caption="Reference Image", width=200)
                                    break
        
        else:
            st.error("Results file not found for this bin")


# ============================================================================
# SECTION 2: PROCESS NEW BIN
# ============================================================================

elif section == "⚡ Process New Bin":
    
    st.markdown('<p class="section-header">Process New Bin: Real-Time Verification</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a bin image and its metadata JSON file to run the complete verification pipeline:
    **Scraping → Detection → CLIP Matching → OCR → Ensemble Validation**
    """)
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader(
            "**Upload Bin Image** 📷",
            type=['jpg', 'jpeg', 'png'],
            help="Upload the bin image you want to verify"
        )
    
    with col2:
        uploaded_metadata = st.file_uploader(
            "**Upload Metadata JSON** 📄",
            type=['json'],
            help="Upload the metadata file containing expected items"
        )
    
    # Process button
    if uploaded_image and uploaded_metadata:
        
        st.markdown("---")
        
        if st.button("🚀 Process Bin", type="primary", use_container_width=True):
            
            # Create temp directory
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded files
            image_path = os.path.join(temp_dir, "uploaded_bin.jpg")
            metadata_path = os.path.join(temp_dir, "uploaded_metadata.json")
            
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            with open(metadata_path, "wb") as f:
                f.write(uploaded_metadata.getbuffer())
            
            # Load metadata
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                st.error(f"❌ Error loading metadata: {e}")
                st.stop()
            
            # Extract ASINs and names
            bin_data = metadata.get('BIN_FCSKU_DATA', {})
            asins_to_scrape = {}
            
            for item in bin_data.values():
                asin = item.get('asin')
                name = item.get('name', 'Unknown Product')
                if asin:
                    asins_to_scrape[asin] = name
            
            if len(asins_to_scrape) == 0:
                st.error("❌ No ASINs found in metadata")
                st.stop()
            
            st.info(f"📦 Found {len(asins_to_scrape)} items to verify")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Scrape reference images
            status_text.markdown("### [1/5] 🕷️ Scraping reference images...")
            progress_bar.progress(20)
            
            ref_folder = "reference_images"
            os.makedirs(ref_folder, exist_ok=True)
            
            try:
                with st.spinner("Scraping Amazon product images..."):
                    scraped_count = scrape_reference_images(
                        asins_to_scrape,
                        ref_folder
                    )
                st.success(f"✅ Scraped {scraped_count}/{len(asins_to_scrape)} reference images")
            except Exception as e:
                st.error(f"❌ Scraping error: {e}")
                st.stop()
            
            # Step 2-5: Load models and process
            status_text.markdown("### [2/5] 🤖 Loading models...")
            progress_bar.progress(40)
            
            matcher, validator = load_models()
            
            if matcher is None or validator is None:
                st.error("❌ Failed to load models")
                st.stop()
            
            # Step 3: Process bin
            status_text.markdown("### [3/5] 🔍 Running CLIP + OCR matching...")
            progress_bar.progress(60)
            
            try:
                matcher_result = matcher.process_bin(
                    image_path,
                    metadata_path,
                    ref_folder
                )
                
                if matcher_result is None:
                    st.error("❌ Processing failed")
                    st.stop()
                
            except Exception as e:
                st.error(f"❌ Matching error: {e}")
                st.stop()
            
            # Step 4: Ensemble validation
            status_text.markdown("### [4/5] 🎯 Creating ensemble results...")
            progress_bar.progress(80)
            
            try:
                ensemble_result = validator.create_ensemble_results(matcher_result)
            except Exception as e:
                st.error(f"❌ Ensemble error: {e}")
                st.stop()
            
            # Step 5: Complete
            status_text.markdown("### [5/5] ✅ Processing complete!")
            progress_bar.progress(100)
            
            st.balloons()
            
            st.markdown("---")
            st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Expected Items", ensemble_result['total_expected'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Detected Items", ensemble_result['total_detected'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Correct Items", ensemble_result['correct_items'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                accuracy = ensemble_result['ensemble_accuracy']
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display results
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("### 📋 Validation Summary")
                
                # Expected items
                st.markdown("**Expected Items:**")
                for item in matcher_result['expected_items']:
                    st.markdown(f"- **{item['asin']}**: {item['name']} (qty: {item.get('quantity', 1)})")
                
                st.markdown("---")
                
                # Missing items
                if len(ensemble_result['missing_items']) > 0:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown("### ❌ Missing/Under-detected Items")
                    for item in ensemble_result['missing_items']:
                        st.markdown(
                            f"- **{item['asin']}**: Expected {item['expected']}, "
                            f"Detected {item['detected']} (Missing {item['missing']})"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Extra items
                if len(ensemble_result.get('extra_items', [])) > 0:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ Extra/Wrong Detections")
                    for item in ensemble_result['extra_items']:
                        st.markdown(f"- **{item['asin']}**: {item['count']} detected (not expected)")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # If all correct
                if ensemble_result['ensemble_accuracy'] == 100.0:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ Perfect Match!")
                    st.markdown("All expected items were correctly detected and matched.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 🖼️ Bin Image with Detections")
                
                # Load and display annotated image
                bin_image = cv2.imread(image_path)
                bin_image_rgb = cv2.cvtColor(bin_image, cv2.COLOR_BGR2RGB)
                
                # Draw bounding boxes
                annotated = bin_image_rgb.copy()
                for match in ensemble_result['ensemble_matches']:
                    x1, y1, x2, y2 = map(int, match.get('bbox', [0, 0, 0, 0]))
                    
                    # Color based on confidence
                    if match['confidence'] == 'HIGH':
                        color = (0, 255, 0)  # Green
                    elif match['confidence'] == 'MEDIUM':
                        color = (255, 165, 0)  # Orange
                    else:
                        color = (255, 0, 0)  # Red
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated, f"Obj {match['object_id']}", 
                               (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                st.image(annotated, use_container_width=True)
            
            st.markdown("---")
            
            # Match details
            st.markdown("### 📊 Detailed Match Results")
            
            for match in ensemble_result['ensemble_matches']:
                
                confidence_color = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🔴'
                }.get(match['confidence'], '⚪')
                
                with st.expander(
                    f"{confidence_color} Object {match['object_id']}: "
                    f"{match['final_asin']} - "
                    f"Confidence: {match['confidence']}"
                ):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**CLIP Results:**")
                        st.markdown(f"- ASIN: {match['clip_asin']}")
                        st.markdown(f"- Score: {match['clip_score']:.3f}")
                    
                    with col2:
                        st.markdown("**OCR Results:**")
                        st.markdown(f"- ASIN: {match['ocr_asin']}")
                        st.markdown(f"- Score: {match['ocr_score']}")
                    
                    with col3:
                        st.markdown("**Final Decision:**")
                        st.markdown(f"- ASIN: {match['final_asin']}")
                        st.markdown(f"- Status: {match['agreement']}")
                    
                    # Show reference image
                    for ext in ['.jpg', '.jpeg', '.png']:
                        ref_path = os.path.join(ref_folder, f"{match['final_asin']}{ext}")
                        if os.path.exists(ref_path):
                            st.image(ref_path, caption=f"Reference: {match['final_asin']}", width=200)
                            break
            
            st.markdown("---")
            
            # Download results
            st.markdown("### 💾 Download Results")
            
            result_json = json.dumps(ensemble_result, indent=2)
            st.download_button(
                label="📥 Download Results JSON",
                data=result_json,
                file_name=f"bin_verification_results.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Please upload both a bin image and metadata JSON file to begin processing")


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray; padding: 2rem;">'
    'Amazon Bin Verification System | Powered by YOLO + CLIP + OCR | Built with Streamlit'
    '</div>',
    unsafe_allow_html=True
)
