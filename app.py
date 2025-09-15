import streamlit as st
import os
import json
import math
import re
from typing import Dict, List, Optional, Tuple
import PyPDF2
import docx
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="ATS Resume Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ATSMatcher:
    """Production-grade ATS matching system for Streamlit"""
    
    def __init__(self, role_family: str = "SWE"):
        self.role_family = role_family
        self.ontology = self.build_ontology(role_family)
        
    def build_ontology(self, role_family="SWE"):
        """Build skill ontology with proper names"""
        base_ontology = {
            "Programming Languages": {
                "terms": ["c++", "cpp", "python", "java", "javascript", "scala", "typescript", "c#"],
                "synonyms": {
                    "c++": ["cpp", "c plus plus"],
                    "javascript": ["js"],
                    "c#": ["csharp", "c sharp"]
                },
                "weight": 0.25,
                "type": "required"
            },
            "Algorithms & Data Structures": {
                "terms": ["algorithms", "data structures", "algorithm", "data structure", "graph theory", "optimization", 
                         "complexity analysis", "sorting", "searching", "trees", "dynamic programming"],
                "synonyms": {
                    "algorithms": ["algorithmic", "algorithm"],
                    "data structures": ["data structure"],
                    "graph theory": ["graph algorithms", "graphs"],
                    "optimization": ["optimize", "optimized", "optimizing"]
                },
                "weight": 0.20,
                "type": "required"
            },
            "Concurrent Programming": {
                "terms": ["multithreading", "concurrency", "parallel processing", "threading", "concurrent programming"],
                "synonyms": {
                    "multithreading": ["multi-threading", "multi threading", "threaded"],
                    "concurrency": ["concurrent"],
                    "parallel processing": ["parallel computing", "parallelization"]
                },
                "weight": 0.15,
                "type": "required"
            },
            "Cloud & DevOps": {
                "terms": ["docker", "kubernetes", "k8s", "aws", "azure", "ci/cd", "jenkins", "containerization"],
                "synonyms": {
                    "kubernetes": ["k8s"],
                    "aws": ["amazon web services"],
                    "ci/cd": ["continuous integration", "continuous deployment"]
                },
                "weight": 0.08,
                "type": "preferred"
            },
            "Machine Learning": {
                "terms": ["machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "xgboost", 
                         "nlp", "computer vision"],
                "synonyms": {
                    "machine learning": ["ml"],
                    "deep learning": ["neural networks"],
                    "scikit-learn": ["sklearn"],
                    "natural language processing": ["nlp"],
                    "computer vision": ["cv"]
                },
                "weight": 0.12,
                "type": "preferred"
            },
            "Version Control": {
                "terms": ["git", "github", "version control"],
                "synonyms": {
                    "git": ["github"],
                    "version control": ["git", "github"]
                },
                "weight": 0.08,
                "type": "required"
            },
            "FPGA/EDA Domain": {
                "terms": ["fpga", "eda", "verilog", "vhdl", "placement", "routing", "timing", "vivado", "quartus"],
                "synonyms": {
                    "fpga": ["field programmable gate array"],
                    "eda": ["electronic design automation"]
                },
                "weight": 0.12,
                "type": "required"
            }
        }
        return base_ontology
    
    def preserve_atomic_tokens(self, text):
        """Preserve atomic tokens like C++"""
        atomic_tokens = {
            'c++': '__CPP__',
            'c#': '__CSHARP__',
            '.net': '__DOTNET__', 
            'node.js': '__NODEJS__',
            'k8s': '__K8S__'
        }

        preserved = text.lower()
        reverse_map = {}

        for token, placeholder in atomic_tokens.items():
            if token in preserved:
                preserved = preserved.replace(token, placeholder)
                reverse_map[placeholder] = token

        return preserved, reverse_map
    
    def normalize_text(self, text):
        """Normalize text while preserving atomic tokens"""
        if not text:
            return ""
            
        preserved, reverse_map = self.preserve_atomic_tokens(text)
        
        normalized = re.sub(r'[^\w\s._-]', ' ', preserved)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Restore atomic tokens
        for placeholder, token in reverse_map.items():
            normalized = normalized.replace(placeholder, token)
            
        return normalized

    def parse_jd(self, jd_text):
        """Parse job description into sections"""
        normalized = self.normalize_text(jd_text)
        lines = [line.strip() for line in normalized.split('\n') if line.strip()]
        
        sections = {
            'required_skills': [],
            'preferred_skills': [],
            'responsibilities': [],
            'education': [],
            'tools_stack': [],
            'soft_factors': []
        }
        
        current_section = 'responsibilities'
        
        section_patterns = {
            'required_skills': re.compile(r'require|must have|essential|mandatory|expected', re.I),
            'preferred_skills': re.compile(r'prefer|nice to have|plus|bonus|desired', re.I),
            'responsibilities': re.compile(r'responsib|duties|role|you will|key tasks', re.I),
            'education': re.compile(r'education|degree|qualification|academic', re.I),
            'tools_stack': re.compile(r'tools|technolog|stack|framework|platform', re.I),
            'soft_factors': re.compile(r'soft skill|communication|leadership|collaboration', re.I)
        }
        
        for line in lines:
            # Detect section changes
            for section, pattern in section_patterns.items():
                if pattern.search(line):
                    current_section = section
                    break
            
            if len(line) > 10 and not re.match(r'^\*+$|^-+$|^=+$', line):
                sections[current_section].append(line)
        
        return sections

    def parse_resume(self, resume_data):
        """Parse resume sections"""
        parsed = {
            'header': '',
            'summary': '',
            'skills': resume_data.get('skills', ''),
            'experience': [],
            'projects': [],
            'education': resume_data.get('education', ''),
            'extras': ''
        }

        # Parse experience
        if resume_data.get('experience'):
            exp_lines = resume_data['experience'].split('\n')
            current_job = {}
            
            for line in exp_lines:
                line = line.strip()
                if ' - ' in line and not line.startswith('-'):
                    if current_job:
                        parsed['experience'].append(current_job)
                    
                    parts = line.split(' - ', 1)
                    current_job = {
                        'company': parts[0].strip(),
                        'title': parts[1].split('(')[0].strip() if len(parts) > 1 else '',
                        'dates': '',
                        'bullets': []
                    }
                elif line.startswith('- '):
                    if current_job:
                        current_job['bullets'].append(line[2:])
            
            if current_job:
                parsed['experience'].append(current_job)

        # Parse projects
        if resume_data.get('projects'):
            project_lines = resume_data['projects'].split('\n')
            for line in project_lines:
                if line.startswith('- ') or line.startswith('• '):
                    parsed['projects'].append(line[2:].strip())
                elif line.strip() and len(line.strip()) > 10:
                    parsed['projects'].append(line.strip())

        return parsed

    def extract_evidence(self, resume, ontology):
        """Extract evidence for each skill bucket"""
        evidence = {}

        for bucket_name, bucket_data in ontology.items():
            bucket_evidence = {
                'score': 0.0,
                'examples': [],
                'missing_terms': [],
                'sections_found': []
            }

            # Collect all resume text
            all_resume_text = self.normalize_text(
                f"{resume['skills']} {' '.join([job.get('bullets', []) for job in resume['experience'] if isinstance(job.get('bullets'), list)])} {' '.join(resume['projects'])} {resume['education']}"
            )

            # Find matching terms
            found_terms = []
            
            # Check main terms
            for term in bucket_data['terms']:
                if term in all_resume_text:
                    found_terms.append({'term': term, 'type': 'exact'})
                    bucket_evidence['examples'].append(f"{term} found in resume")

            # Check synonyms
            if 'synonyms' in bucket_data:
                for main_term, synonyms in bucket_data['synonyms'].items():
                    for synonym in synonyms:
                        if synonym in all_resume_text and not any(f['term'] == main_term for f in found_terms):
                            found_terms.append({'term': main_term, 'type': 'synonym', 'matched': synonym})
                            bucket_evidence['examples'].append(f"{main_term} (via {synonym}) found in resume")

            # Calculate evidence score
            if found_terms:
                coverage = len(found_terms) / len(bucket_data['terms'])
                multiplier = 1.0
                
                # Bonuses
                if len(found_terms) >= 2:
                    multiplier += 0.2
                
                exact_matches = sum(1 for f in found_terms if f['type'] == 'exact')
                if exact_matches > 0:
                    multiplier += 0.1
                
                bucket_evidence['score'] = min(coverage * multiplier * 1.2, 1.0)

            # Mark missing terms only if no evidence
            if not found_terms:
                bucket_evidence['missing_terms'] = bucket_data['terms'][:3]

            evidence[bucket_name] = bucket_evidence

        return evidence

    def score_resume_to_jd(self, jd_text, resume_data):
        """Main scoring function"""
        ontology = self.build_ontology("SWE")
        jd = self.parse_jd(jd_text)
        parsed_resume = self.parse_resume(resume_data)
        evidence = self.extract_evidence(parsed_resume, ontology)

        # Scoring weights
        weights = {
            'RCS': 35, 'RRA': 20, 'PSM': 15, 'EDU': 10, 'EES': 10, 'KC': 5, 'ATS': 5
        }

        # 1. Required Core Skills (35%)
        rcs_score = 0.0
        required_buckets = [(k, v) for k, v in ontology.items() if v['type'] == 'required']
        total_required_weight = sum(data['weight'] for _, data in required_buckets)
        
        for bucket, data in required_buckets:
            if evidence[bucket]['score'] > 0:
                adjusted_score = max(evidence[bucket]['score'], 0.7)
                rcs_score += (data['weight'] / total_required_weight) * adjusted_score
        rcs_score *= weights['RCS']

        # 2. Preferred Skills Match (15%)
        psm_score = 0.0
        preferred_buckets = [(k, v) for k, v in ontology.items() if v['type'] == 'preferred']
        total_preferred_weight = sum(data['weight'] for _, data in preferred_buckets)
        
        if total_preferred_weight > 0:
            for bucket, data in preferred_buckets:
                if evidence[bucket]['score'] > 0:
                    psm_score += (data['weight'] / total_preferred_weight) * math.sqrt(evidence[bucket]['score'])
        psm_score *= weights['PSM']

        # 3. Responsibilities Alignment (20%)
        all_bullets = []
        for job in parsed_resume['experience']:
            if job.get('bullets'):
                all_bullets.extend(job['bullets'])
        all_bullets.extend(parsed_resume['projects'])

        rra_score = 0.6  # Base score
        action_words = ['implement', 'develop', 'build', 'design', 'optimize', 'create', 'deploy']
        tech_words = ['algorithm', 'system', 'application', 'framework', 'model', 'pipeline']
        
        combined_text = ' '.join(all_bullets).lower()
        if any(word in combined_text for word in action_words):
            rra_score += 0.15
        if any(word in combined_text for word in tech_words):
            rra_score += 0.15
        
        rra_score = min(rra_score, 1.0) * weights['RRA']

        # 4. Education (10%)
        edu_text = self.normalize_text(parsed_resume['education'])
        degree_match = 0.0
        if re.search(r'master|m\.s|ms\b', edu_text, re.I):
            degree_match = 1.0
        elif re.search(r'bachelor|b\.s|bs\b', edu_text, re.I):
            degree_match = 0.8

        field_match = 0.0
        if re.search(r'computer.*(?:science|engineering)|electrical.*engineering', edu_text, re.I):
            field_match = 1.0
        elif re.search(r'engineering', edu_text, re.I):
            field_match = 0.7
        
        edu_score = (0.6 * degree_match + 0.4 * field_match) * weights['EDU']

        # 5. Evidence Strength (10%)
        metric_matches = len(re.findall(r'\d+%|\d+x', ' '.join(all_bullets)))
        metric_ratio = metric_matches / max(len(all_bullets), 1)
        ees_raw = 1 - math.exp(-2 * metric_ratio)
        ees_score = ees_raw * weights['EES']

        # 6. Keyword Coverage (5%)
        jd_words = self.normalize_text(jd_text).split()
        resume_text = self.normalize_text(' '.join([str(v) for v in resume_data.values()]))
        jd_bigrams = [f"{jd_words[i]} {jd_words[i+1]}" for i in range(len(jd_words)-1) if len(jd_words[i]) > 2]
        matched = sum(1 for bigram in jd_bigrams if bigram in resume_text)
        kc_score = min(matched / max(len(jd_bigrams), 1), 0.9) * weights['KC']

        # 7. ATS Hygiene (5%)
        ats_raw = 0.0
        if len(parsed_resume['education']) > 20: ats_raw += 0.2
        if len(parsed_resume['experience']) > 0: ats_raw += 0.2
        if len(parsed_resume['skills']) > 30: ats_raw += 0.2
        if len(parsed_resume['projects']) > 0: ats_raw += 0.2
        if metric_matches >= 2: ats_raw += 0.2
        ats_score = ats_raw * weights['ATS']

        # Calculate total
        total_score = rcs_score + psm_score + rra_score + edu_score + ees_score + kc_score + ats_score

        # Generate gaps
        critical_gaps = []
        moderate_gaps = []
        
        for bucket, data in ontology.items():
            if evidence[bucket]['score'] == 0:
                gap = {'bucket': bucket, 'reason': 'no evidence found', 'impact': data['weight']}
                if data['type'] == 'required':
                    critical_gaps.append(gap)
                else:
                    moderate_gaps.append(gap)

        # Apply clamp if critical gaps
        explainability = {'must_have_clamp_applied': False, 'impact_bonus': 0.0}
        if critical_gaps:
            total_score = min(total_score, 80.0)
            explainability['must_have_clamp_applied'] = True

        # Impact bonus
        if ees_raw > 0.5:
            bonus = 2.0
            total_score = min(total_score + bonus, 100.0)
            explainability['impact_bonus'] = bonus

        return {
            'score_overall': round(total_score, 1),
            'subscores': {
                'RCS': round(rcs_score, 1), 'RRA': round(rra_score, 1),
                'PSM': round(psm_score, 1), 'EDU': round(edu_score, 1),
                'EES': round(ees_score, 1), 'KC': round(kc_score, 1), 'ATS': round(ats_score, 1)
            },
            'buckets': [
                {
                    'name': name, 'type': data['type'], 'evidence': round(evidence[name]['score'], 2),
                    'weight': data['weight'], 'examples': evidence[name]['examples'][:3],
                    'missing_terms': evidence[name]['missing_terms'][:5]
                }
                for name, data in ontology.items()
            ],
            'critical_gaps': sorted(critical_gaps, key=lambda x: x['impact'], reverse=True),
            'moderate_gaps': sorted(moderate_gaps, key=lambda x: x['impact'], reverse=True),
            'explainability': explainability
        }


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX {docx_path}: {e}")
        return ""

def parse_resume_text_to_sections(text):
    """Parse raw resume text into structured sections using regex and heuristics"""
    if not text:
        return {"education": "", "experience": "", "skills": "", "projects": ""}
    
    # Normalize text
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
    
    sections = {"education": "", "experience": "", "skills": "", "projects": ""}
    
    # Define section patterns (case insensitive)
    section_patterns = {
        'education': r'(?i)(?:education|academic|qualification|degree|university|college|school)',
        'experience': r'(?i)(?:experience|employment|work|professional|career|jobs?)',
        'skills': r'(?i)(?:skills?|technical|competenc|technolog|programming|tools?)',
        'projects': r'(?i)(?:projects?|portfolio|work samples?|personal projects?)'
    }
    
    # Split text into lines
    lines = text.split('\n')
    current_section = None
    section_content = {key: [] for key in sections.keys()}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line is a section header
        section_found = None
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line) and len(line) < 50:  # Headers are usually short
                section_found = section_name
                break
        
        if section_found:
            current_section = section_found
            continue
        
        # Add content to current section
        if current_section:
            section_content[current_section].append(line)
        else:
            # If no section detected, try to categorize by content
            line_lower = line.lower()
            if any(word in line_lower for word in ['university', 'college', 'degree', 'bachelor', 'master', 'phd']):
                section_content['education'].append(line)
            elif any(word in line_lower for word in ['company', 'engineer', 'developer', 'manager', 'analyst']):
                section_content['experience'].append(line)
            elif any(word in line_lower for word in ['python', 'java', 'c++', 'javascript', 'programming']):
                section_content['skills'].append(line)
            elif any(word in line_lower for word in ['project', 'built', 'developed', 'created']):
                section_content['projects'].append(line)
    
    # Convert lists back to strings
    for section_name in sections.keys():
        sections[section_name] = '\n'.join(section_content[section_name])
    
    return sections

def convert_document_to_json(file_path):
    """Convert PDF/DOCX to structured JSON resume format"""
    file_extension = Path(file_path).suffix.lower()
    
    # Extract text based on file type
    if file_extension == '.pdf':
        raw_text = extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        raw_text = extract_text_from_docx(file_path)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
    if not raw_text:
        st.error(f"Could not extract text from {file_path}")
        return None
    
    # Parse into sections
    sections = parse_resume_text_to_sections(raw_text)
    
    return sections

def load_resumes_from_folder():
    """Load resume files from resumes folder - supports PDF, DOCX, JSON, TXT"""
    resumes_folder = "resumes"
    resume_files = {}
    
    if not os.path.exists(resumes_folder):
        st.error(f"Resumes folder '{resumes_folder}' not found. Please create it and add resume files.")
        return {}
    
    supported_extensions = ['.pdf', '.docx', '.doc', '.json', '.txt']
    
    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            continue
        
        try:
            with st.spinner(f"Processing {filename}..."):
                if file_extension == '.json':
                    # Load JSON directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        resume_data = json.load(f)
                        
                elif file_extension == '.txt':
                    # Load text file (assume sections separated by double newlines)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        sections = content.split('\n\n')
                        resume_data = {
                            'education': sections[0] if len(sections) > 0 else '',
                            'experience': sections[1] if len(sections) > 1 else '',
                            'skills': sections[2] if len(sections) > 2 else '',
                            'projects': sections[3] if len(sections) > 3 else ''
                        }
                        
                elif file_extension in ['.pdf', '.docx', '.doc']:
                    # Convert document to JSON
                    resume_data = convert_document_to_json(file_path)
                    
                    # Save as JSON for future use
                    json_filename = Path(filename).stem + '.json'
                    json_path = os.path.join(resumes_folder, json_filename)
                    
                    if resume_data and not os.path.exists(json_path):
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(resume_data, f, indent=2, ensure_ascii=False)
                        st.success(f"Converted {filename} → {json_filename}")
                
                if resume_data:
                    resume_files[filename] = resume_data
                    
        except Exception as e:
            st.warning(f"Error processing {filename}: {e}")
    
    return resume_files


def main():
    """Main Streamlit app"""
    st.title("🎯 ATS Resume Matcher")
    st.markdown("**Production-grade resume-job description matching with actionable insights**")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Configuration")
        
        # Add file upload option
        st.markdown("**Option 1: Upload New Resume**")
        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX resume:",
            type=['pdf', 'docx', 'doc'],
            help="Upload a resume file to analyze"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Convert to JSON
            with st.spinner(f"Processing {uploaded_file.name}..."):
                resume_data = convert_document_to_json(temp_path)
                
            # Clean up temp file
            os.remove(temp_path)
            
            if resume_data:
                st.success(f"Successfully processed {uploaded_file.name}")
                selected_resume_data = resume_data
                selected_resume_file = uploaded_file.name
                
                # Show preview
                with st.expander("Preview Extracted Resume"):
                    for section, content in resume_data.items():
                        if content:
                            st.text_area(f"{section.title()}:", content, height=80, key=f"upload_{section}")
            else:
                st.error("Failed to process uploaded file")
                selected_resume_data = None
                selected_resume_file = None
        else:
            selected_resume_data = None
            selected_resume_file = None
        
        st.markdown("**Option 2: Select from Existing Resumes**")
        
        # Load resumes from folder
        resumes = load_resumes_from_folder()
        
        if not resumes and not uploaded_file:
            st.warning("No resumes found. Please upload a file or add resume files to the 'resumes' folder.")
            st.info("Supported formats: PDF, DOCX, JSON, TXT")
        
        # Resume selection dropdown (only if no file uploaded)
        if not uploaded_file and resumes:
            selected_resume_file = st.selectbox(
                "Select Resume:",
                options=list(resumes.keys()),
                help="Choose from available resume files"
            )
            selected_resume_data = resumes.get(selected_resume_file)
            
            # Show preview
            if selected_resume_file:
                st.success(f"Selected: {selected_resume_file}")
                with st.expander("Preview Selected Resume"):
                    for section, content in selected_resume_data.items():
                        if content:
                            st.text_area(f"{section.title()}:", content, height=80, key=f"folder_{section}")
    
    with col2:
        st.subheader("📄 Job Description")
        
        # JD input box
        jd_text = st.text_area(
            "Paste Job Description:",
            height=400,
            help="Paste the complete job description here",
            placeholder="Paste job description text here..."
        )
        
        # Analyze button
        if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
            if not jd_text.strip():
                st.error("Please enter a job description")
            elif not selected_resume_data:
                st.error("Please upload a resume or select one from the folder")
            else:
                with st.spinner("Analyzing resume-JD match..."):
                    # Run analysis
                    matcher = ATSMatcher()
                    result = matcher.score_resume_to_jd(jd_text, selected_resume_data)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Overall score
                    score = result['score_overall']
                    if score >= 80:
                        score_color = "green"
                    elif score >= 60:
                        score_color = "orange"
                    else:
                        score_color = "red"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #{score_color}20; border: 2px solid {score_color};">
                        <h1 style="color: {score_color}; margin: 0;">{score}%</h1>
                        <h3 style="color: {score_color}; margin: 0;">Overall ATS Match</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Missing Skills
                    st.subheader("❌ Missing Skills")
                    all_gaps = result['critical_gaps'] + result['moderate_gaps']
                    
                    if not all_gaps:
                        st.success("No critical gaps found!")
                    else:
                        for gap in all_gaps:
                            is_critical = gap in result['critical_gaps']
                            gap_color = "red" if is_critical else "orange"
                            status = "Required" if is_critical else "Preferred"
                            
                            st.markdown(f"""
                            <div style="padding: 10px; margin: 5px 0; border-left: 4px solid {gap_color}; background-color: #{gap_color}10;">
                                <strong>{gap['bucket']}</strong> ({status}) - <em>{gap['reason']}</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Optional: Show component scores
                    with st.expander("📊 Detailed Score Breakdown"):
                        st.json(result['subscores'])
                    
                    # Explanations
                    if result['explainability']['must_have_clamp_applied']:
                        st.warning("⚠️ Score capped at 80% due to missing required skills")
                    
                    if result['explainability']['impact_bonus'] > 0:
                        st.info(f"✅ +{result['explainability']['impact_bonus']} bonus for strong quantified results")


if __name__ == "__main__":
    main()