from pydantic import BaseModel, Field
from typing import Optional, Iterator, Callable

import streamlit as st
from pypdf import PdfReader

from phi.workflow import Workflow, RunResponse, RunEvent

from phi.tools.jina_tools import JinaReaderTools
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.huggingface import HuggingFaceChat
from phi.model.deepseek import DeepSeekChat
from phi.model.openrouter import OpenRouter
from phi.model.mistral import MistralChat
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.log import logger
from phi.utils.pprint import pprint_run_response

from dotenv import load_dotenv
import os
from jobspy import scrape_jobs
import json

# Model definitions (Gemini and Groq are used in this example)
MODEL_GEMINI: Gemini = Gemini(id="gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_GROQ: Groq = Groq(id="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

# Removed unused model definitions (commented out)

class LoggerManager:
    def __init__(self, logger, main_status_placeholder=None):
        self.logger = logger
        self.main_status_placeholder = main_status_placeholder

    def log(self, message):
        self.logger.info(message)
        if self.main_status_placeholder:
            self.main_status_placeholder.write(message)


class JobOffer(BaseModel):
    title: str = Field(..., description="Title of the Job Offer.")
    company: str = Field(..., description="Company.")
    description: Optional[str] = Field(..., description="Description of the Job Offer.")

class JobOffers(BaseModel):
    offers: list[JobOffer] = Field(..., description="List of Job Offers.")

class Keyword(BaseModel):
    keyword: str = Field(..., description="Keyword, it can be either single word or an expression")

class KeywordFrequency(BaseModel):
    keyword: str = Field(..., description="Keyword, it can be either single word or an expression")
    frequency: int = Field(..., description="How many job offers listed this keyword")

class ListKeywordsFrequency(BaseModel):
    keywords: list[KeywordFrequency] = Field(..., description="List of Keywords.")

class ListKeywords(BaseModel):
    keywords: list[str] = Field(..., description="List of Keywords. It can contains repated keywords.")

def get_latest_jobs_for_profession(profession: str):
    """
    This function searches for the latest job offers for a specific profession.

    Args:
        profession (str): Desired profession for job openings.
        location (str): Desired location for jobs. Defaults to empty string.

    Returns:
        list: List of dictionaries containing job offer information.
    """

    # Defines the sites to be searched
    sites = ["indeed", "linkedin", "glassdoor", "google"]

    jobs = scrape_jobs(
        site_name=sites,
        search_term=profession,
        location="",
        results_wanted=15,
        hours_old=168
    )

    if jobs is not None and not jobs.empty:
        jobs = jobs[["company", "title", "description"]]
        jobs = jobs.fillna("")
        jobs_list = jobs.to_dict(orient="records")  # Converts to a list of dictionaries
        # logger_manager.log(f"Jobs list: {jobs_list}")  # Log for debugging

        # Creates a list of JobOffer objects
        job_offers = []
        for job_dict in jobs_list:
            try:
                # Ensures that all required fields are present
                if all(key in job_dict for key in ["company", "title", "description"]):
                    job_offers.append(JobOffer(
                        title=str(job_dict["title"]),
                        company=str(job_dict["company"]),
                        description=str(job_dict["description"])
                    ))
            except Exception as e:
                logger.error(f"Error creating JobOffer: {e}")
                continue

        if job_offers:
            return JobOffers(offers=job_offers).model_dump_json()

    logger.warning("No valid job offers found")
    return JobOffers(offers=[])

class RecommendationGenerator(Workflow):
    extractor_job_offer = Agent(tools=[JinaReaderTools(api_key=os.getenv("JINA_API_KEY"))], model=MODEL_GROQ,
            description=f"You are a famous leadership expert in HR Analysts. You excel at extracting titles, skills, keywords from job offers.",
            instructions=[f"Given a job offer, extract as many keywords as possible related specifically to the profession. ",
                          "Search for skills (human related or technical), methodologies and technologies (including tools), etc."
                          "Ensure the keywords are related to the description of the role itself and not about the company or about the benefits. ",
                          "The result should be a list of keywords in the response model format."],
            response_model=ListKeywords, markdown=True))
    

    extractor: Agent = Agent(model=MODEL_GROQ,
            description=f"You are a famous leadership expert in HR Analysts. You excel at extracting titles, skills, keywords from job offers.",
            instructions=[f"Given a list of job offers, extract as many keywords as possible related specifically to the profession. ",
                          "Search for profession titles, skills (human related or technical), methodologies and technologies (including tools), etc."
                          "Iterate over each job offer and add their extracted keywords in the final list. That is, the list could contain repeated keywords from different job offers (not from the same job offer), ignoring the job offer they were extracted from."
                          "Ensure the keywords are related to the description of the role itself and not about the company or about the benefits. ",
                          "The result should be a list of keywords in the response model format."],
            response_model=ListKeywords, markdown=True)

    prioritizer: Agent = Agent(model=MODEL_GROQ,
            description=f"You excel at evaluating keywords and the frequency with which they occur. You are accurate and do not fabricate data.",
            instructions=[f"Evaluate the list of keywords and tell me the frequency of each one",
                          "Sort the keywords by the frequency in descending order",
                          "Exclude the keywords with only one occurrence (frequency)",
                          "The result should be in the response model format."],
            response_model=ListKeywordsFrequency, markdown=True)

    recommender: Agent = Agent(model=MODEL_GEMINI,
                description="You are a famous HR analyst. You are very good at suggesting improvements to resumes based on a list of keywords and best practices for resume writing.",
                instructions=["The resume points you should evaluate: Headline, Summary, Work Experience (experience title and experience summary)",
                    "Be specific about which section above you will make a suggestion for, show how it currently is, and how you suggest changing it based on the given keywords and relate it with the context of the person's experience.",
                    "Review the details of each company mentioned in the person's experience, evaluating all keywords. Give a final suggestion for each experience, reasoning and relating the keywords",
                    "Each suggestion of keywords should be contextual and only suggest if infered that the person has this experience, skill or knowledge, otherwise create a separate suggestion confirming if the person has this experience",
                    "When evaluating the title and summary of the resume take into account the whole experience of the person in each company, so you can suggest most contextual keywords and justify the link with specific experiences.",
                    "Always use this topics evaluation each section: Current, Detailed Reasoning (mentioning keywords), Suggested Text - Alternative 1, Suggested Text - Alternative 2",
                    "You don't need to explain the evaluation structure to the user.",
                    "Your result should always be in the same language of the resume"])

    def run(self, profession: str, curriculum_vitae_file_name: str, curriculum_vitae_content: str, use_cache: bool = True) -> RunResponse:
        logger_manager.log(f"🚀 Getting the last job offers and generating recommendations: {profession}")
        sorted_keywords = None

        if use_cache and "cached_profession" in self.session_state:
            logger_manager.log("🗄️ Checking if keywords from job offers exists in cache")
            for cached_profession in self.session_state["cached_profession"]:
                if "profession" in cached_profession and cached_profession["profession"] == profession:
                    logger_manager.log("🗄️ Found cached keywords")
                    sorted_keywords = cached_profession["sorted_keywords"]

        search_results: Optional[JobOffers] = None
        if not sorted_keywords:
            logger_manager.log("🗄️  No cached keywords found.")
            logger_manager.log("🚀  Search the job offers related to profession.")
            # Step 1: Search the job offers related to profession
            num_tries = 0
            # Run until we get a valid job offers
            while search_results is None and num_tries < 3:
                try:
                    num_tries += 1

                    logger_manager.log(f"🚀 Searching job offers for {profession}")
                    jobs = get_latest_jobs_for_profession(profession)
                    searcher_response = RunResponse(content=jobs)

                    if searcher_response and searcher_response.content:
                        search_results = JobOffers.model_validate_json(searcher_response.content)
                    else:
                        search_results = None
                        logger.warning("🔍 Searcher response invalid, trying again...")
                except Exception as e:
                    logger.warning(f"Error running searcher: {e}")

            # If no search_results are found for the topic, end the workflow
            if search_results is None or len(search_results.offers) == 0:
                return RunResponse(
                    run_id=self.run_id,
                    event=RunEvent.workflow_completed,
                    content=f"Sorry, could not find any job offer related to profession: {profession}",
                )

            # Step 2: Extract keywords in batches
            logger_manager.log("🚀 Extracting keywords from job offers")

            job_offers = [v.model_dump() for v in search_results.offers]
            all_keywords = [] # Initialize an empty list to accumulate keywords

            for i in range(0, len(job_offers), 3):
                batch_offers = job_offers[i:i+3]
                extractor_input = {
                    "profession": profession,
                    "offers": batch_offers,
                }

                logger_manager.log(f"🚀 Extracting keywords from job offers batch {i // 3 + 1}")
                keywords_response: RunResponse = self.extractor.run(json.dumps(extractor_input, indent=4))

                # Accumulate keywords
                if keywords_response.content and keywords_response.content.keywords:
                    all_keywords.extend(keywords_response.content.keywords)

            # Step 3: Sorting the keywords
            logger_manager.log("🗂️ Sorting the keywords")
            prioritizer_input = {
                "keywords": all_keywords
            }
            sorted_keywords_response: RunResponse = self.prioritizer.run(json.dumps(prioritizer_input, indent=4))
            logger_manager.log("✅ Sorting the keywords...DONE")

            if sorted_keywords_response.content and sorted_keywords_response.content.keywords:
                sorted_keywords = sorted_keywords_response.content
            else:
                sorted_keywords = []

            if "cached_profession" not in self.session_state:
                self.session_state["cached_profession"] = []
            self.session_state["cached_profession"].append({"profession": profession, "sorted_keywords": sorted_keywords.model_dump()})
            logger_manager.log(f"🗄️ Saved in the cache")

        # Step 4: Extract keywords from a specific link
        keywords_job_offer = ListKeywords([keywords=[]])
        if link_job_offer:
            logger_manager.log(f"🚀 Extracting keywords from specific job offer in {link_job_offer}")
            extractor_input = {
                    "profession": profession,
                    "offer": f"read the link of the job offer: {link_job_offer}",
                }
            keywords_job_offer_response: RunResponse = self.extractor_job_offer.run(json.dumps(extractor_input, indent=4))
            if keywords_job_offer_response.content and keywords_job_offer_response.content.keywords:
                keywords_job_offer = keywords_job_offer_response.content.keywords

        # Step 5: Generating recommendation
        logger_manager.log("✍️ Generating recommendation")
        keywords = sorted_keywords.get('keywords', []) if isinstance(sorted_keywords, dict) else [v.model_dump() for v in sorted_keywords.keywords]
        recommender_input = {
            "curriculum_vitae_content": curriculum_vitae_content,
            ""
            "keywords_frequency": keywords
        }

        # Run the writer and yield the response
        recommendation: RunResponse = self.recommender.run(json.dumps(recommender_input, indent=4), stream=False, markdown=True)
        logger_manager.log("✅ Generating recommendation: DONE")
        return recommendation

load_dotenv()

st.title("Resume Evaluator")

# Main content area
with st.expander("How does it work?", expanded=False):
    st.markdown("""
    ### How does it work?

    Before you begin, here's a summary of how the system works to ensure you know what to expect:
    """)

    st.markdown("""
    1. **Job Portal Search:** The system searches for vacancies on more than 3 job portals to find contextual keywords.
    2. **Keyword Extraction:** The vacancies found are analyzed to extract relevant keywords for the profession.
    3. **Keyword Prioritization:** The extracted keywords are evaluated and prioritized based on the frequency of occurrence.
    4. **Recommendation Generation:** Based on the prioritized keywords, the system generates specific recommendations to improve the resume.
    """)

# Sidebar for inputs
with st.sidebar:
    st.header("Inputs")
    profession = st.text_input("For which profession do you want to refine your resume?", key="profession_input")
    curriculo_pdf = st.file_uploader("Upload your resume in PDF format", type=["pdf"], key="curriculo_pdf_input")
    curriculo_file_name = ""
    curriculo_conteudo = ""
    
    # Checkbox to force new job search
    search_new_jobs = st.checkbox("Always search for new jobs instead of using the cache")
    
    # Check if both required fields are filled
    is_button_disabled = not (profession and curriculo_pdf)
    # if not profession:
    #     st.warning("Please enter a profession")
    # if not curriculo_pdf:
    #     st.warning("Please upload a PDF resume")
        
    if st.button("Evaluate resume", disabled=is_button_disabled, use_container_width=True):
        st.session_state['run_evaluation'] = True
    else:
        st.session_state['run_evaluation'] = False

# Status and results area
sidebar_status_placeholder = st.empty()
main_status_placeholder = st.empty()
logger_manager = LoggerManager(logger, main_status_placeholder)

if curriculo_pdf:
    curriculo_file_name = curriculo_pdf.name
    with st.sidebar:
        sidebar_status_placeholder.write("PDF resume uploaded successfully!")
        sidebar_status_placeholder.write("Converting PDF to text...")

    try:
        pdf_reader = PdfReader(curriculo_pdf)
        for page in pdf_reader.pages:
            curriculo_conteudo += page.extract_text() + "\n\n"
        with st.sidebar:
            st.success("PDF converted to text successfully!")
    except Exception as e:
        with st.sidebar:
            sidebar_status_placeholder.write"Error processing the PDF: {e}")
        st.stop()

if st.session_state.get('run_evaluation', False):
    # Create the workflow
    generate_recommendation = RecommendationGenerator(
        session_id=f"recommendation-to-{profession}",
        storage=SqlWorkflowStorage(
            table_name="recommendation_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    main_status_placeholder.empty()

    # Run workflow
    recommendation: RunResponse = generate_recommendation.run(
        profession=profession,
        curriculum_vitae_file_name=curriculo_file_name,
        curriculum_vitae_content=curriculo_conteudo,
        use_cache=not search_new_jobs
    )

    main_status_placeholder.empty()  # Clear the status placeholder
    st.markdown(recommendation.content)
