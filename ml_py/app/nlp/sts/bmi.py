import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

MIN_SENTENCE_LEN = 20
BATCH_LEN = 32


model: SentenceTransformer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_bmi(job, resume):
    """
    job: list of strings. Each string is a single sentence, cleaned before BERT tokenization
    resume: list of string. Each string is a single sentence, cleaned before BERT tokenization
    return: dict of the form:
    {
        'bmi': float,
        'matches': list of dicts of the form:
        {
            'job': string: sentence from job description
            'resume': string: sentence from resume
            'match: float: % match
        }
    }
    """
    # Preprocessing
    job, resume = clean_texts(job, resume)

    # Text to integer tokens
    job_tokens = [model.tokenize(sentence) for sentence in job]
    resume_tokens = [model.tokenize(sentence) for sentence in resume]

    # Int tokens to embeddings
    job_embeddings = get_token_embeddings(job_tokens)
    resume_embeddings = get_token_embeddings(resume_tokens)

    # Token embeddings to sentence embeddings (Mean of all token embeddings)
    job_embeddings = get_sentence_embeddings(job_embeddings)
    resume_embeddings = get_sentence_embeddings(resume_embeddings)

    cos_scores = get_cos_scores(job_embeddings, resume_embeddings)

    scores = get_scores_map(cos_scores, job, resume)
    bmi = calculate_bmi(scores)

    return bmi, scores[:10]


def get_token_embeddings(tokens):
    return model.encode(
        tokens,
        batch_size=BATCH_LEN,
        output_value="token_embeddings",
        is_pretokenized=True,
        device=device,
    )


def get_sentence_embeddings(token_embeddings):
    embeddings = []
    for embedding in token_embeddings:
        mask = embedding != 0
        sentence_embedding = (embedding * mask).sum(axis=0) / mask.sum(axis=0)
        embeddings.append(sentence_embedding)
    return torch.tensor(embeddings)


def calculate_bmi(scores):
    scores = [score["match"] for score in scores]
    top_n = 5
    top_n = top_n if len(scores) >= top_n else len(scores)

    avg = sum(scores[:top_n]) / top_n * 100
    return round(avg, 2)


def get_scores_map(cos_scores, job, resume):
    cos, max_args = cos_scores.max(1)

    scores = []

    for job_i in range(len(job)):
        scores.append(
            {
                "job": job[job_i],
                "resume": resume[int(max_args[job_i])],
                "match": float(cos[job_i]),
                "job_idx": job_i,
                "resume_idx": max_args[job_i],
            }
        )

    return sorted(scores, key=lambda s: -s["match"])


def get_cos_scores(job_embeddings, resume_embeddings):

    cos_scores = util.pytorch_cos_sim(job_embeddings, resume_embeddings)
    cos_scores = sigmoid(cos_scores, a=9.8, b=1.017, c=5.8)

    return cos_scores


def init_model():
    global model
    if model is None:

        # List of pre-trained models: https://www.sbert.net/docs/pretrained_models.html
        # These will be downloaded from https://sbert.net/models/<model_name>.zip
        # The model will be cached in ~/.cache/torch/sentence_transformers/sbert.net_models_<model_name>

        model_name = "distilbert-base-nli-stsb-mean-tokens"  # small <250 MB RAM
        # model_name = 'roberta-base-nli-stsb-mean-tokens'      # medium
        # model_name = 'roberta-large-nli-stsb-mean-tokens'  # Large 1.3 GB RAM

        model = SentenceTransformer(model_name, device=device)


def remove_before_word(sentence, word):
    try:
        subject_index = sentence.index(word)
        sentence = sentence[subject_index + len(word) :]
    except:
        # print(f"HEY! No word '{word}' in {sentence[:30]}...")
        pass
    return sentence


def is_valid_sentence(sentence):
    if len(sentence) < MIN_SENTENCE_LEN:
        return False
    if " " not in sentence:
        return False
    return True


def clean_and_split_resume(resume):
    resume = basic_cleanup(resume)

    resume = remove_before_word(resume, "SUMMARY:")
    resume = remove_before_word(resume, "{{END_DOCUMENT_DATESTAMP}}")

    sentences = re.split(r"[\n\.]", resume)

    sentences = [sentence for sentence in sentences if is_valid_sentence(sentence)]

    return sentences


def clean_and_split_job(job):
    job = basic_cleanup(job)

    sentences = re.split(r"[\n\.]", job)

    sentences = [sentence for sentence in sentences if is_valid_sentence(sentence)]

    return sentences


def basic_cleanup(text):
    text = re.sub(r"<br[ ]*/>", "\n", text)  # Replace <br/> with \n
    text = re.sub(r'<[a-z_\-,;#0-9"\'\\=: A-Z]+>', "", text)  # Remove opening HTML tag
    text = re.sub(
        r"</[a-z]+>", ". ", text
    )  # Replace closing HTML tag with a period and space.
    text = re.sub(
        r"\.[ ]*\.", ".", text
    )  # Remove additional periods caused by above step
    text = re.sub(r"&nbsp;", " ", text)  # Remove nbsp
    text = re.sub(r"&rsquo;", "", text)  # Remove nbsp
    text = re.sub(r"[ ]+", " ", text)  # Remove whitespaces

    return text


def sigmoid(x, a=1.0, b=1.0, c=0.0):
    return b / (1 + np.exp(-x * a + c))


def clean_texts(job, resume):
    try:
        job = clean_and_split_job(job)
        resume = clean_and_split_resume(resume)
        print("cleaned texts")

        if len(job) == 0 or len(resume) == 0:
            raise Exception("Job or resume is too short")
        return job, resume
    except Exception as e:
        print(f"Exception in cleaning texts: {e}")


def main():
    init_model()

    resume = "\n\n{{HEADER}}\n\n{{END_HEADER}}\n\n\n{{FOOTER}}\n\n\n\n{{END_FOOTER}}\n\n{{DOCUMENT_DATESTAMP}} 2019-04-03 {{END_DOCUMENT_DATESTAMP}}\nDivya Kuthuru\nE: divya.tad2233@gmail.com\nC: (614) 495-7014\nColumbus, OH\n\nSUMMARY\n\n* Expertise in Software Development Life Cycle (SDLC) and Test Development Life Cycle (TDLC) in Agile as well as Iterative development environments.\n* Experience in Test Plan Development, System test, Functional, Integration, Regression and UAT testing in Agile Environment.\n* Experience in preparing Test plan, Test scenarios, Test cases and Test summary reports for both automated and manual testing based on User requirements, System requirements and Use case documents.\n* Experience in handling Keyboard and Mouse Events, Accessing Forms, tables, and link, using Selenium Web Driver.\n* Experience in handling multiple windows, Alerts and Pop-ups with Selenium.\n* Expertise in Grouping of Test Cases, Test Methods and Test Suites for regression and functional testing using the TestNG annotations like Groups, Parameter, Data Provider.\n* Experience in Selenium Automation using Selenium Web Driver, Selenium IDE, Java, Cucumber BDD, TestNG and POM (Page Object Model) framework.\n* Used Selenium WebDriver and TestNG to run parallel testing by creating Groups and categorizing test cases.\n* Experienced in using Maven build tools to manage framework dependency jar files.\n* Experienced in Web Services testing using Postman and database testing.\n* Experience in validating request and response XML, SOAP and RESTFUL Web service calls.\n* Experienced in using Apache POI to read data from external sources to feed locators into the test script.\n* Experience in different Version Control Systems and continuous integration and deployment tools (GitHub, Jenkins).\n* Developed Test Scripts to implement Test Cases, Test Scenarios, and features for BDD (Behavior Driven Development) using Cucumber.\n* Experience in Database Testing using SQL Queries\n* Expertise in different types of testing like Automation testing, Manual testing, Integration testing, System testing, Smoke testing, Regression testing, JUnit Testing, Black box testing, Functional testing, Database testing, GUI testing, Web / UI and (UAT)User Acceptance Testing.\n* Expertise in working knowledge of Core Java and Object-Oriented Concepts (Inheritance, Polymorphism, Exception Handling, Multi-Threading and Collections).\n* Experienced in using Test Management tools such as RTC, QC, JIRA to track test progress, execution and deliverables.\n* Hands on experience in using build and project management tools like MAVEN.\n* Good team player with the ability to manage and work independently.\n\nTECHNOLOGY\n\n Test Approaches Waterfall, Agile/Scrum, SDLC, STLC, Bug Life Cycle\n Testing Tools Selenium WebDriver, Tricentis TOSCA, TestNG, Selenium IDE, JUnit, Cucumber, Gherkin, SoapUI, Sauce Labs, HP QTP/UFT\n\n Test Build& Integration Tools Maven, Jenkins\n Frameworks Cucumber (BDD), Page Object Model (POM), TestNG, Junit, Keyword Driven, Data Driven, and Hybrid framework.\n\n Programming Tools JAVA, C, C++, HTML, XML, JSON\n Markup Languages HTML, XML, XPath, CSS Selector\n Databases MySQL, Oracle, SQL Server\n Browsers Internet Explorer, IE Edge, Mozilla Firefox, Google Chrome, Safari\n Operating Systems Windows 7/8, Ubuntu, UNIX, LINUX\n Defect Tools RTC, HP Quality Center, JIRA\n MS Office Tools Outlook, Word, Excel, PowerPoint, MS Access\n Utilities Eclipse, GIT, Firebug, Fire Path and RTC&RRC\n\n\nPROFESSIONAL EXPERIENCE\n\nAbercrombie & Fitch, New Albany, OH Nov 2018 - Present Automation Engineer/QA Analyst\n\n* Used Page Object Model (POM) for Functional and Regression testing using Selenium, TestNG, Maven and Java.\n* Proficient with testing REST APIs, Web & Database testing\n* Responsible for writing test automation scripts for execution of all the test cases including web and mobile automation.\n* Used grouping for the test cases to run against Smoke tests, Regression Suite and Integration Test cases.\n* Responsible for integrating the automation scripts with Jenkins for Continuous Integration (CI) automation testing.\n* Responsible for triaging daily test automation failures on daily builds.\n* Experience on Writing SQL queries to extract data from various source tables to perform database testing.\n* Used Postman for rest service testing\n* Responsible for planning test activities, preparing test scenarios and test cases and prioritize testing activities based on the requirement of the project.\n* Responsible for maintaining test cases in Quality Center and tracking defects in JIRA board.\n* Responsible for preparing the document with Test automation framework enhancements and provide it to the team.\n* Extensively used Git hub repository to pull/push the automation code.\n* Responsible for prioritizing the test cases during the release testing activities.\n* Responsible for Carrying out retesting every time when changes are made to the code to fix defects.\n* Used sauce labs for parallel execution.\n* Performed System, Integration, Smoke, Functional, End to End, Positive and Negative and monitored the behavior of the applications during different phases of testing using testing methodologies.\n\nEnvironment: Selenium IDE, WebDriver, IntelliJ, JIRA, TestNG, MySQL, Postman, Java, HP Quality Center, Jenkins, Git.\n\nPNC Bank, Pittsburgh, PA - https://www.google.com/search?rlz=1C1GCEV_enUS825US825&q=Pittsburgh&stick=H4sIAAAAAAAAAOPgE-LUz9U3MMlLK09S4gAxzSwKjLS0spOt9POL0hPzMqsSSzLz81A4VhmpiSmFpYlFJalFxQCumyjVQwAAAA&sa=X&ved=2ahUKEwje3MTljYLgAhXi24MKHfZxADYQmxMoATAjegQICBAL{{END_DX_TEXT_TO_DELETE}} Sep 2017 - Oct 2018 ;\nQA Analyst/Test Automation Developer\n\n* Involved in creation of Test plan and responsible for creating Test cases from the functional requirements.\n* Developed scenarios and scenario outlines in feature files of Cucumber BDD using Gherkin language.\n* Performed automation of test cases for Regression Testing using Cucumber BDD and Selenium WebDriver, Maven for testing Functional validations.\n* Created and executed automation scripts and performed Functional and Regression Testing for various releases.\n* Executed automated Selenium scripts and reproduced failures manually.\n* Performed mobile application testing manually and using simulator.\n* Used automated scripts and performed functionality testing during the various phases of the application development using Cucumber BDD framework.\n* Used SQL queries to perform database testing.\n* Created and Verified Web services API requests\n* Expertise on performing API testing using Postman.\n* Involved in creating Test Case Scenarios, Test Case Execution and maintaining defects using JIRA Tool.\n* Logged and tracked defects using the tool JIRA.\n* Developed and executed complex SQL Queries and Procedures to perform database testing.\n* Participated in Sprint planning, Sprint retrospective and daily scrum meetings.\n\nEnvironment: Java, Selenium Web Driver, IntelliJ, SQL, HTML, XML, Cucumber, Jenkins, SoapUI, Postman, Maven.\n\nTarget, Minneapolis, MN Sep 2016 - Aug 2017\nQA Analyst/ Test Automation Developer\n\n* Worked in a highly dynamic Agile environment.\n* Participated in designing and implementing different automation framework from scratch like POM (Page Object Model) framework using Selenium WebDriver, TestNG Data Provider and Apache POI API.\n* Performing Regression Testing on the new builds as the new feature enhancements or fixes are merged.\n* Updating the test scenario and adjusting automation scripts accordingly as the requirements or features change from sprint to sprint.\n* Appling TestNG and Extent Report for generating graphical reports for clients as need.\n* Connected to database to Query the database using SQL for data verification and validation.\n* Maintaining git repositories and branches for automation source code in GitHub and performing merge, pull and resolving conflicts.\n* Logging new issues and updating status of existing tickets using Defect Tracking & Management in Jira.\n* Diligently working with multiple projects without compromising quality of testing or project deadline.\n* Participating with peers for knowledge sharing sessions, mentoring junior members and always extending helping hands to other members if they fall behind.\n\nEnvironment: Java, Selenium Web Driver, Eclipse, SQL Server, HTML, XML, JUnit, TestNG, Jenkins.\n\nCognizant Technology Solutions, India Jun 2015 - July 2016\nProgrammer Analyst\n\n* Assessed & analyzed user stories and participated in Sprint planning, Review Sessions & Scrum Meetings and developed Test scenarios, Test cases, Test data, Test reports.\n* Managed individual sprints user stories and tasks using JIRA as tracking tool and agile tool.\n* Developed and Executed Test Scripts using Java, Selenium WebDriver, TestNG, analyzed Test Results.\n* Performed Functional testing as per user stories and performed Integration Testing & System Testing using Selenium WebDriver automation scripts.\n* Handled Keyboard and Mouse Events, Accessing Forms, tables, and link, using Selenium Web Driver.\n* Handled multiple windows, Alerts and Pop-ups with Selenium.\n* Handled in testing with handling different methods of Select class for selecting and deselecting for drop down using Selenium.\n* Designed and Implemented Data Driven Framework and extracted data from external Excel files using Apache POI and loaded into the variables in the scripted code.\n* Configured the Test Cases to receive input Test Data Sets for the corresponding test cases using TestNG Data Provider Annotation.\n* Used FireBug, FirePath to debug, edit and locate the objects based on ID, Name, XPath, CSS Selector, Link, Partial Link, Attributes and Tags\n* Performed BDD (Behavior Driven Development) using Cucumber Features, Scenarios and Step Definitions in Gherkin format.\n* Managed the framework dependency jars using Maven.\n* Involved in Web services testing using SOAPUI Tool.\n* Managed the project build and the project dependencies using Maven.\nEnvironment: Java, Selenium WebDriver, Selenium IDE, Jenkins, SoapUI, HTML, XML, JIRA, Microsoft Office, Cucumber, Eclipse, Maven, Agile, Gherkin, SQL and RTC.\n\nSonata Software, India April 2014 - May 2015\nQA Tester\n* Involved in Requirement gathering, Requirement analysis, Design, Development, Integration and Deployment.\n* Responsible for working within a team to create document and execute Test Plan, Test Cases, and test scripts.\n* Developed Manual test scenarios and test cases from Business Requirements and Design Documents.\n* Attended scrum meetings as per Agile methodology.\n* Prepared Manual Test Scenarios and Test Cases for the Application under test.\n* Participated in co-ordination and implementation of QA methodology.\n* Prepared Test Data and executed Test Cases from Quality Center.\n* Perform static, functional, technical integration, end-to-end and User Acceptance Testing\n* Performed backend testing for data validation using SQL Queries.\n* Performed rigorous m Functional Testing, Smoke testing, Integration testing, UAT Testing, Backend Testing, Regression Testing, End to End Testing and System Testing.\n* Tested critical bug fixes and coordinated with developers in release of bug fixes meeting tight timeline.\n* Analyze and make specific recommendations on improvements that can be integrated into business processes.\n* Reported Daily about the team progress to the Project Manager and Team Lead.\n\nEnvironment: JAVA, Eclipse, Manual Testing, Selenium, Agile, SQL, Windows\n\nEDUCATION\n\nEastern Illinois University \t May 2018\nMaster's in computer technology\n\nVasavi College of Engineering \t May 2015\nB.E in Electrical and Electronics Engineering\n\n"
    desc = "<p><strong>Please submit all candidates at your best Competitive Rate!<br /><br />GENERAL FUNCTION: </strong></p> <p>As part of a cross-functional Agile development team, the SDET&rsquo;s primary role is to ensure quality through delivery of test automation best practices.&nbsp; Responsibilities include developing new test automation code using various automation tools and frameworks, maintaining existing test automation code with a high degree of quality, leveraging strong software design and test automation principles, and using personal ingenuity and creativity to find new ways to test software solutions.</p> <p>The SDET works hand-in-hand with all members of the Agile team to determine customer and product requirements and their applicability to product code and test code. They also perform manual testing when needed. They must continually partner to openly exchange ideas and opinions, elevate concerns, and follow quality policies and procedures as defined by Fifth Third&rsquo;s quality governance standards.<br /><br />This specific role will focus on creation and execution of both manual and automated scripts focused on API and service-level validations.&nbsp;</p> <p><strong>ESSENTIAL RESPONSIBILITIES:</strong></p> <p><strong>Own- </strong>Own and be accountable for test strategy, test planning, and test execution within the product team</p> <p><strong>Code</strong>- Implement high-quality, reusable, maintainable test automation code using various automation tools and frameworks.</p> <p><strong>Triage</strong>- Efficiently reproduce reported software problems, analyze data, and work with other team members to quickly remove obstacles.</p> <p><strong>Problem Solve</strong>- Resolve difficult issues, often times with little information, spanning across multiple applications.</p> <p><strong>Develop and Organize Tests</strong>- Create structured, clean, and cohesive test cases for all new features and/or functional changes in the software, organized into repeatable test suites.</p> <p><strong>Document</strong>- Ensure testing activities always grow collective team knowledge through strong documentation and notes.</p> <p><strong>Build Expertise</strong>- Build a deep understanding of Fifth Third products and services while sharing testing knowledge and best practices with all members of the Agile team.</p> <p><strong>Research-</strong> Study emerging test tools, trends, and methodologies that will enhance existing systems and processes.</p> <p><strong>KEY SKILLS: </strong></p> <ul> <li>5 years&rsquo; experience in Quality Assurance and Testing</li> <li>Experience developing automation frameworks and test suites</li> <li>Programming experience in Javascript and/or Java</li> <li>2 years&rsquo; experience working in an Agile/Scrum environment</li> <li>Experience with front-end automated testing in Selenium Webdriver or an equivalent</li> <li>Experience with application life cycle management tools such as Jenkins, GIT, Version1, Quality Center</li> <li><strong>API tools: SoapUI, Postman, or similar</strong></li> <li><strong>Experience and strong skillset in testing application services, API, SOAP, REST</strong></li> </ul> <p><strong>SUPERVISORY RESPONSIBILITIES: </strong>None.</p> <p><strong>MINIMUM KNOWLEDGE, SKILLS AND ABILITIES REQUIRED: </strong></p> <p>Undergraduate degree or equivalent with 3-6 years of experience.</p>"

    bmi = get_bmi(desc, resume)
    print(bmi)


if __name__ == "__main__":
    main()
