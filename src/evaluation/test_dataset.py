"""
test_dataset.py — Curated evaluation dataset for AI Act Navigator RAGAS assessment

28 question/ground_truth pairs covering all major AI Act compliance scenarios.
Ground truths are derived directly from the official AI Act text (EU 2024/1689).

Dataset structure:
  - question:       the query sent to the retrieval pipeline
  - ground_truth:   the correct answer grounded in the AI Act
  - category:       query type for stratified analysis
  - articles:       primary articles the answer should reference
  - difficulty:     easy / medium / hard
  - retrieval_hint: which strategy should excel (dense/sparse/hybrid)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    category: str
    articles: list[str]
    difficulty: str = "medium"
    retrieval_hint: str = "hybrid"
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "articles": self.articles,
            "difficulty": self.difficulty,
            "retrieval_hint": self.retrieval_hint,
        }


# ---------------------------------------------------------------------------
# Test dataset — 28 samples across 5 categories
# ---------------------------------------------------------------------------

EVAL_DATASET: list[EvalSample] = [

    # =========================================================
    # CATEGORY 1: CLASSIFICATION (8 samples)
    # =========================================================

    EvalSample(
        sample_id="cls_01",
        question="What AI practices are prohibited under the AI Act?",
        ground_truth=(
            "Article 5 of the AI Act prohibits several AI practices: "
            "(1) AI systems that deploy subliminal techniques to manipulate persons against their will; "
            "(2) systems that exploit vulnerabilities of specific groups (age, disability) to distort behaviour; "
            "(3) biometric categorisation systems inferring sensitive attributes like race or political opinions; "
            "(4) social scoring by public authorities; "
            "(5) real-time remote biometric identification in public spaces for law enforcement (with narrow exceptions); "
            "(6) AI used to infer emotions in workplaces or educational institutions; "
            "(7) AI that creates or expands facial recognition databases through untargeted scraping."
        ),
        category="classification",
        articles=["Art. 5"],
        difficulty="easy",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="cls_02",
        question="What are the two conditions that must both be met for an AI system to be classified as high-risk under Article 6(1)?",
        ground_truth=(
            "Under Article 6(1), an AI system is high-risk when it meets two cumulative conditions: "
            "(1) it is intended to be used as a safety component of a product, or is itself a product, "
            "covered by Union harmonisation legislation listed in Annex II; AND "
            "(2) that product is required to undergo a third-party conformity assessment under that legislation. "
            "Both conditions must be satisfied simultaneously."
        ),
        category="classification",
        articles=["Art. 6(1)", "Annex II"],
        difficulty="medium",
        retrieval_hint="sparse",
    ),

    EvalSample(
        sample_id="cls_03",
        question="Which Annex III categories cover AI systems used in education and employment?",
        ground_truth=(
            "Annex III lists high-risk AI systems in eight domains. "
            "Point 3 covers education and vocational training: systems that determine access to educational "
            "institutions, assess students, or monitor for prohibited behaviour during tests. "
            "Point 4 covers employment and workers management: systems used for recruitment, "
            "CV screening, promotion decisions, task allocation, and performance monitoring. "
            "Both categories are considered high-risk due to their significant impact on individuals' "
            "life opportunities and fundamental rights."
        ),
        category="classification",
        articles=["Annex III", "Art. 6(2)"],
        difficulty="easy",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="cls_04",
        question="Can a provider self-assess that their Annex III system is not high-risk?",
        ground_truth=(
            "Yes. Article 6(3) allows providers to assess that an AI system listed in Annex III "
            "does not pose a significant risk of harm to health, safety or fundamental rights, "
            "and therefore is not high-risk. This assessment must be documented before placing "
            "the system on the market or putting it into service. "
            "The provider must register the system in the EU database under Article 49(2) "
            "and make the documentation available to national competent authorities upon request. "
            "The Commission must provide guidelines on the practical implementation of this provision "
            "by 2 February 2026."
        ),
        category="classification",
        articles=["Art. 6(3)", "Art. 49(2)"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="cls_05",
        question="Is a conversational AI assistant used in mental health support high-risk?",
        ground_truth=(
            "A mental health support AI assistant may be classified as high-risk depending on its context. "
            "If it is used in a healthcare setting and makes or influences decisions about access to "
            "health services, it could fall under Annex III point 5 (essential private and public services). "
            "If it targets vulnerable users such as minors or people with disabilities, "
            "Article 9(9) requires providers to give special consideration to adverse impacts on these groups. "
            "However, if the system is purely informational and advisory without influencing access decisions, "
            "it may fall under limited risk with Art. 50 transparency obligations only. "
            "A precautionary Art. 6(3) self-assessment is recommended."
        ),
        category="classification",
        articles=["Annex III", "Art. 6(2)", "Art. 6(3)", "Art. 50"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="cls_06",
        question="What AI systems are listed in Annex III under the law enforcement category?",
        ground_truth=(
            "Annex III point 6 lists the following law enforcement AI systems as high-risk: "
            "(a) systems for individual risk assessments to predict reoffending or criminal activity; "
            "(b) polygraph and similar tools to detect deception; "
            "(c) systems to assess reliability of evidence in criminal proceedings; "
            "(d) systems to predict the occurrence or recurrence of criminal offences based on profiling; "
            "(e) systems for profiling of natural persons in the course of criminal detection or investigation; "
            "(f) AI for crime analytics to identify unknown patterns or relationships."
        ),
        category="classification",
        articles=["Annex III point 6"],
        difficulty="medium",
        retrieval_hint="sparse",
    ),

    EvalSample(
        sample_id="cls_07",
        question="Does the AI Act apply to AI systems developed or used exclusively for military or national security purposes?",
        ground_truth=(
            "No. Article 2(3) explicitly excludes AI systems developed or used exclusively for "
            "military, national security, or defence purposes from the scope of the AI Act, "
            "regardless of the type of entity carrying out those activities. "
            "This exclusion applies to both public and private actors when acting in these contexts."
        ),
        category="classification",
        articles=["Art. 2(3)"],
        difficulty="medium",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="cls_08",
        question="What is the definition of a general-purpose AI model under the AI Act?",
        ground_truth=(
            "Article 3(63) defines a general-purpose AI model (GPAI model) as an AI model that "
            "displays significant generality and is capable of competently performing a wide range "
            "of distinct tasks regardless of the way the model is placed on the market, "
            "and that can be integrated into a variety of downstream systems or applications. "
            "This includes large generative models trained on large amounts of data using self-supervision "
            "at scale. It does NOT include AI models used for research, development or prototyping "
            "activities before they are placed on the market."
        ),
        category="classification",
        articles=["Art. 3(63)", "Art. 51"],
        difficulty="medium",
        retrieval_hint="sparse",
    ),

    # =========================================================
    # CATEGORY 2: OBLIGATIONS — HIGH RISK (8 samples)
    # =========================================================

    EvalSample(
        sample_id="obl_01",
        question="What are the requirements for a risk management system for high-risk AI systems?",
        ground_truth=(
            "Article 9 requires providers of high-risk AI systems to establish, implement, document "
            "and maintain a risk management system throughout the system's lifecycle. "
            "It must be a continuous iterative process comprising: "
            "(a) identification and analysis of known and foreseeable risks to health, safety or fundamental rights; "
            "(b) estimation and evaluation of risks including under foreseeable misuse; "
            "(c) evaluation of risks from post-market monitoring data; "
            "(d) adoption of appropriate risk management measures. "
            "Testing must be performed at appropriate points during development and prior to market placement. "
            "Providers must give special consideration to impacts on persons under 18 and other vulnerable groups."
        ),
        category="obligations_high_risk",
        articles=["Art. 9"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="obl_02",
        question="What data governance requirements apply to training data for high-risk AI systems?",
        ground_truth=(
            "Article 10 requires that training, validation and testing data for high-risk AI systems "
            "must be subject to appropriate data governance practices including: "
            "examination for possible biases; identification of relevant data gaps or shortcomings; "
            "appropriate measures to ensure sufficient statistical properties including "
            "representativeness with respect to persons or groups of persons; "
            "compliance with EU and national law on intellectual property and privacy. "
            "Special categories of personal data may be processed only where strictly necessary "
            "to ensure bias monitoring, detection and correction in high-risk AI systems."
        ),
        category="obligations_high_risk",
        articles=["Art. 10"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="obl_03",
        question="What information must be included in the technical documentation for a high-risk AI system?",
        ground_truth=(
            "Article 11 requires providers to draw up technical documentation before placing "
            "a high-risk AI system on the market. The content is specified in Annex IV and includes: "
            "a general description of the system and its intended purpose; "
            "description of elements and development process; "
            "information on the training methodology and techniques; "
            "validation and testing procedures and results; "
            "the risk management system documentation; "
            "description of changes made throughout the lifecycle; "
            "a list of standards applied; "
            "the EU declaration of conformity. "
            "Documentation must be kept for 10 years after the system is placed on the market."
        ),
        category="obligations_high_risk",
        articles=["Art. 11", "Annex IV"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="obl_04",
        question="What human oversight measures must be built into high-risk AI systems?",
        ground_truth=(
            "Article 14 requires high-risk AI systems to be designed and developed with human oversight measures "
            "that allow natural persons to effectively oversee the system during its use. "
            "These measures must enable overseers to: "
            "(a) fully understand the capacities and limitations of the system; "
            "(b) monitor operations and detect signs of anomalies, dysfunctions and unexpected performance; "
            "(c) disregard, override or reverse the output of the system; "
            "(d) intervene or interrupt the system via a 'stop' button or similar procedure. "
            "For systems requiring prior authorisation for use, oversight must be performed by a natural person "
            "with the necessary competence, training and authority."
        ),
        category="obligations_high_risk",
        articles=["Art. 14"],
        difficulty="medium",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="obl_05",
        question="What are the obligations of deployers of high-risk AI systems?",
        ground_truth=(
            "Article 26 sets out deployer obligations for high-risk AI systems: "
            "use systems in accordance with instructions for use; "
            "assign human oversight to competent natural persons; "
            "ensure input data is relevant and representative for the intended purpose; "
            "monitor the operation of the system on the basis of instructions for use; "
            "inform providers about serious incidents; "
            "keep logs generated by the system for a minimum period. "
            "Deployers that are public bodies must also conduct a fundamental rights impact assessment "
            "before deploying high-risk AI systems listed in Annex III."
        ),
        category="obligations_high_risk",
        articles=["Art. 26", "Art. 27"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="obl_06",
        question="What does Article 9 paragraph 9 require regarding vulnerable groups?",
        ground_truth=(
            "Article 9(9) requires that when implementing the risk management system, "
            "providers shall give consideration to whether in view of its intended purpose "
            "the high-risk AI system is likely to have an adverse impact on persons under the age of 18 "
            "and, as appropriate, other vulnerable groups. "
            "This consideration must inform the risk management measures adopted under Article 9(2)(d). "
            "This provision specifically recognises that children and other vulnerable populations "
            "may face heightened risks from AI systems and deserve explicit protective attention."
        ),
        category="obligations_high_risk",
        articles=["Art. 9(9)"],
        difficulty="hard",
        retrieval_hint="sparse",
    ),

    EvalSample(
        sample_id="obl_07",
        question="What quality management system obligations apply to providers of high-risk AI systems?",
        ground_truth=(
            "Article 17 requires providers of high-risk AI systems to put a quality management system in place. "
            "The quality management system must cover at minimum: "
            "a strategy for regulatory compliance; "
            "techniques and processes for system design; "
            "system development and quality control; "
            "examination, test and validation procedures; "
            "technical standards to be applied; "
            "data management systems; "
            "the risk management system; "
            "post-market monitoring; "
            "incident reporting procedures; "
            "handling of communication with authorities. "
            "Documentation must be systematic and orderly in the form of written policies and procedures."
        ),
        category="obligations_high_risk",
        articles=["Art. 17"],
        difficulty="medium",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="obl_08",
        question="When must high-risk AI systems be registered in the EU database?",
        ground_truth=(
            "Article 49 requires providers to register high-risk AI systems in the EU database "
            "before placing them on the market or putting them into service. "
            "For high-risk AI systems listed in Annex III, registration must occur before market placement. "
            "Providers who assess under Art. 6(3) that their system is not high-risk must also register, "
            "but in a separate section of the database. "
            "Deployers of high-risk AI systems intended for use by public authorities "
            "must also register in the database. "
            "The EU database is publicly accessible, with the exception of information "
            "whose disclosure would jeopardise public interests."
        ),
        category="obligations_high_risk",
        articles=["Art. 49"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    # =========================================================
    # CATEGORY 3: TRANSPARENCY — ART. 50 (4 samples)
    # =========================================================

    EvalSample(
        sample_id="tra_01",
        question="What transparency obligations apply to AI chatbots and conversational systems?",
        ground_truth=(
            "Article 50(1) requires providers of AI systems intended to interact directly with natural persons "
            "to ensure that those persons are informed they are interacting with an AI system, "
            "unless this is obvious from the context or circumstances of use. "
            "This obligation applies at the time of first interaction. "
            "The information must be provided in a clear and distinguishable manner. "
            "This requirement is the primary obligation for conversational AI systems like chatbots "
            "that do not otherwise fall into a higher risk tier."
        ),
        category="transparency",
        articles=["Art. 50(1)"],
        difficulty="easy",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="tra_02",
        question="What are the transparency obligations for AI systems that generate synthetic content like deepfakes?",
        ground_truth=(
            "Article 50(4) requires persons who use AI systems to generate or manipulate image, audio or video "
            "content constituting a deep fake to disclose that the content has been artificially generated "
            "or manipulated. This disclosure must be made in a manner that is clearly visible or audible. "
            "An exception applies when the use is for legitimate purposes such as the exercise of freedom "
            "of expression or in the context of authorised security testing. "
            "Article 50(2) also requires providers of AI systems generating synthetic audio, image, video "
            "or text content to ensure their outputs are marked in a machine-readable format "
            "and detectable as artificially generated."
        ),
        category="transparency",
        articles=["Art. 50(2)", "Art. 50(4)"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="tra_03",
        question="Do transparency obligations under Article 50 apply to high-risk AI systems as well?",
        ground_truth=(
            "Yes. Article 50 transparency obligations can apply in addition to high-risk obligations. "
            "Article 50(1) on AI interaction disclosure and Article 50(3) on emotion recognition systems "
            "apply regardless of whether a system is also classified as high-risk under Article 6. "
            "High-risk systems already subject to Article 13 (transparency and information provision) "
            "must comply with both the Art. 13 requirements for deployers "
            "and the Art. 50 requirements for end-user disclosure. "
            "The obligations are complementary, not mutually exclusive."
        ),
        category="transparency",
        articles=["Art. 50", "Art. 13"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="tra_04",
        question="What must providers disclose about AI systems that use emotion recognition?",
        ground_truth=(
            "Article 50(3) requires providers and deployers of AI systems that perform emotion recognition "
            "to inform natural persons exposed to such systems of the system's operation. "
            "This obligation applies at the time of and before exposure to the system. "
            "The information must be provided in a clear and distinguishable manner. "
            "Additionally, Article 5(1)(f) prohibits emotion recognition systems used in workplaces "
            "and educational institutions, which is a separate prohibition from the transparency obligation."
        ),
        category="transparency",
        articles=["Art. 50(3)", "Art. 5(1)(f)"],
        difficulty="medium",
        retrieval_hint="dense",
    ),

    # =========================================================
    # CATEGORY 4: GPAI (4 samples)
    # =========================================================

    EvalSample(
        sample_id="gpai_01",
        question="What transparency obligations apply to providers of GPAI models?",
        ground_truth=(
            "Article 53 requires providers of GPAI models to: "
            "(a) draw up and maintain technical documentation including training process, "
            "data used, testing and evaluation results; "
            "(b) provide information and documentation to downstream providers; "
            "(c) establish a policy to respect EU copyright law including the text and data mining exception; "
            "(d) publish a summary of training data used. "
            "These obligations apply from 2 August 2025. "
            "The required content for technical documentation is specified in Annexes XI and XII."
        ),
        category="gpai",
        articles=["Art. 53", "Annex XI", "Annex XII"],
        difficulty="medium",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="gpai_02",
        question="What additional obligations apply to GPAI models with systemic risk?",
        ground_truth=(
            "Article 55 imposes additional obligations on providers of GPAI models with systemic risk: "
            "(a) perform model evaluations including adversarial testing; "
            "(b) assess and mitigate systemic risks including their sources; "
            "(c) keep track of, document and report serious incidents and corrective measures; "
            "(d) ensure an adequate level of cybersecurity protection. "
            "A GPAI model is presumed to have systemic risk when trained using total compute "
            "greater than 10^25 FLOPs (Article 51(2)). "
            "The AI Office may also classify models as having systemic risk based on other criteria."
        ),
        category="gpai",
        articles=["Art. 55", "Art. 51"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="gpai_03",
        question="Does a company that builds an application using a GPAI model like GPT-4 need to comply with GPAI obligations?",
        ground_truth=(
            "No. Title V GPAI obligations (Articles 51-56) apply to providers of GPAI models, "
            "not to downstream deployers or application builders. "
            "A company that integrates GPT-4 into their application is a deployer of a GPAI-based system, "
            "not a GPAI model provider. "
            "The GPAI compliance obligations fall on OpenAI as the model provider. "
            "However, the downstream application provider must still comply with obligations "
            "applicable to their specific use case (e.g., high-risk obligations if the application "
            "is high-risk, or Art. 50 transparency if it is a chatbot)."
        ),
        category="gpai",
        articles=["Art. 51", "Art. 53", "Art. 3(63)"],
        difficulty="hard",
        retrieval_hint="dense",
    ),

    EvalSample(
        sample_id="gpai_04",
        question="When did GPAI model obligations enter into force?",
        ground_truth=(
            "GPAI model obligations under Title V (Articles 51-56) entered into application "
            "on 2 August 2025, twelve months after the AI Act entered into force on 1 August 2024. "
            "Providers of GPAI models placed on the market after this date must comply immediately. "
            "Providers of GPAI models already on the market before 2 August 2025 "
            "must comply by 2 August 2027. "
            "The GPAI Code of Practice published in July 2025 provides practical guidance "
            "on how providers can fulfil these obligations."
        ),
        category="gpai",
        articles=["Art. 113", "Art. 51"],
        difficulty="medium",
        retrieval_hint="sparse",
    ),

    # =========================================================
    # CATEGORY 5: CROSS-REFERENCE CHAINS (4 samples)
    # =========================================================

    EvalSample(
        sample_id="xref_01",
        question="Article 9 refers to post-market monitoring. What does the referenced article require?",
        ground_truth=(
            "Article 9(2)(c) references the post-market monitoring system referred to in Article 72. "
            "Article 72 requires providers of high-risk AI systems to establish and document "
            "a post-market monitoring system that actively and systematically collects, documents "
            "and analyses data on performance of high-risk AI systems throughout their lifetime. "
            "The monitoring plan must be part of the technical documentation. "
            "For AI systems intended for use by consumers, post-market monitoring must also integrate "
            "and analyse feedback from the serious incidents reporting procedure."
        ),
        category="cross_reference",
        articles=["Art. 9(2)(c)", "Art. 72"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="xref_02",
        question="What does Annex IV specify about technical documentation for high-risk AI systems?",
        ground_truth=(
            "Annex IV specifies the required content of technical documentation under Article 11. "
            "It requires: a general description of the AI system including its intended purpose, "
            "version and date; description of hardware, software components and development processes; "
            "description of training methodologies, techniques and training data sets used; "
            "information on validation and testing procedures, metrics used and test results; "
            "a copy of the EU declaration of conformity; "
            "detailed description of the system's capabilities, limitations, accuracy and robustness; "
            "description of the human oversight measures including technical means to facilitate oversight; "
            "description of any pre-determined changes and their impact on performance."
        ),
        category="cross_reference",
        articles=["Annex IV", "Art. 11"],
        difficulty="medium",
        retrieval_hint="sparse",
    ),

    EvalSample(
        sample_id="xref_03",
        question="Article 13 requires transparency information — who is the intended recipient of this information?",
        ground_truth=(
            "Article 13 requires that high-risk AI systems be designed and developed in a way that "
            "ensures sufficient transparency to enable deployers to interpret the output and use it appropriately. "
            "The information under Article 13 is primarily addressed to deployers, not end users. "
            "It must be provided in the instructions for use and include: "
            "the identity and contact details of the provider; "
            "the characteristics, capabilities and limitations of the system; "
            "information on the input data required; "
            "the level of accuracy, robustness and cybersecurity; "
            "human oversight measures needed. "
            "This is distinct from Article 50 transparency which is directed at the natural persons "
            "interacting with the system."
        ),
        category="cross_reference",
        articles=["Art. 13", "Art. 50"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),

    EvalSample(
        sample_id="xref_04",
        question="What is a fundamental rights impact assessment and who must conduct it?",
        ground_truth=(
            "Article 27 requires deployers of high-risk AI systems listed in Annex III "
            "that are bodies governed by public law, or private entities providing public services, "
            "to conduct a fundamental rights impact assessment (FRIA) before deployment. "
            "The FRIA must assess the risks to fundamental rights that the use of the system may produce "
            "and include: a description of the processes in which the system will be used; "
            "the period of use; categories of natural persons affected; "
            "specific risks of harm to fundamental rights; "
            "human oversight measures; "
            "actions taken to address identified risks. "
            "The assessment must be notified to the relevant market surveillance authority."
        ),
        category="cross_reference",
        articles=["Art. 27", "Annex III"],
        difficulty="hard",
        retrieval_hint="hybrid",
    ),
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_dataset() -> list[EvalSample]:
    """Return the full evaluation dataset."""
    return EVAL_DATASET


def get_by_category(category: str) -> list[EvalSample]:
    """Return samples for a specific category."""
    return [s for s in EVAL_DATASET if s.category == category]


def get_by_difficulty(difficulty: str) -> list[EvalSample]:
    """Return samples for a specific difficulty level."""
    return [s for s in EVAL_DATASET if s.difficulty == difficulty]


def dataset_summary() -> dict:
    """Return summary statistics for the dataset."""
    categories = {}
    difficulties = {}
    hints = {}
    for s in EVAL_DATASET:
        categories[s.category] = categories.get(s.category, 0) + 1
        difficulties[s.difficulty] = difficulties.get(s.difficulty, 0) + 1
        hints[s.retrieval_hint] = hints.get(s.retrieval_hint, 0) + 1
    return {
        "total": len(EVAL_DATASET),
        "by_category": categories,
        "by_difficulty": difficulties,
        "by_retrieval_hint": hints,
    }


if __name__ == "__main__":
    summary = dataset_summary()
    print(f"Evaluation dataset: {summary['total']} samples")
    print(f"\nBy category:")
    for cat, n in summary["by_category"].items():
        print(f"  {cat:30s}: {n}")
    print(f"\nBy difficulty:")
    for diff, n in summary["by_difficulty"].items():
        print(f"  {diff:10s}: {n}")
    print(f"\nBy retrieval hint:")
    for hint, n in summary["by_retrieval_hint"].items():
        print(f"  {hint:10s}: {n}")
