# Biomedical Knowledge Graph

Information extraction from unstructured text to build an interactive knowledge graph.

## Project Description

The aim of the project is to build an interactive knowledge graph of biomedical entities and relations by extracting structured information from unstructured text on drug repurposing for COVID-19 published in biomedical research papers. 

This infectious disease, caused by the SARS-CoV-2 virus, could be considered the greatest public health emergency of international concern thus far in the 21st century. The World Health Organisation declared the outbreak a pandemic on 11 March, 2020, and governments announced lockdown measures constituting the largest quarantine in human history. Inevitably, antiviral drug repurposing activities were accelerated. 

In addition to predicting or recommending new purposes for existing drugs, thereby reducing time, cost and risk, other use cases might include predicting drug side or adverse effects. 

## Data Collection

Data will be collected from multiple sources including [Europe PMC](https://europepmc.org/) and [arXiv](https://arxiv.org/) using APIs to retrieve article full text XML and metadata, [GROBID](https://github.com/kermitt2/grobid) machine learning library to extract, parse and restructure articles in PDF format into structured XML, and more manual methods for page layout analysis, and detecting and extracting text where exceptions occur during automated processes.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is an iterative process with multiple stages, types, tools and techniques, the overall objective being to obtain actionable insights which can inform data-driven decision-making. 
Basic EDA will be performed to analyse text statistics, followed by Word Analysis which will use spaCy to analyse stopwords and n-grams. Outputs will be visualised as histograms, bar charts and wordclouds.

NLP approaches have evolved over time from traditional rule-based to statistical machine learning methods and feature-based approaches, to shallow and deep learning neural network-based models, pre-trained transformers and LLMs, a combination of which will be used for preprocessing, feature extraction and modelling.

spaCy processing pipeline components will be utilised for preprocessing, which facilitates feature extraction and modelling by making natural language normalised and machine-readable.

Feature extraction using traditional frequency-based methods will consider the Bag of Words model as the simplest text vectorisation approach, and represent and identify the importance of words with categorical (CountVectorizer) and weighted (TF-IDF) word representation techniques, and the use of n-grams.

Representation learning, also known as feature learning, simplifies raw input data by extracting meaningful and informative feature representations that are low-dimensional and generalisable. Techniques will include Dimensionality Reduction (PCA, SVD, TruncatedSVD, t-SNE, and UMAP), Word Embeddings (Word2Vec and GloVe) and Text Embeddings (BERT-based Sentence Transformers and GPT-style OpenAI embedding models).

![Visualisation of relationships for "covid-19", "pandemic", "drug", "disease" and "remdesivir"](https://github.com/alisonmitchell/Biomedical-Knowledge-Graph/blob/main/02_Exploratory_Data_Analysis/images/glove_network_explorer.png?raw=true)

Modelling will utilise the general-purpose, scalable K-means algorithm for the downstream task of clustering, a practical application for embeddings after dimensionality reduction. Topic modelling, or abstracting topics from a collection of documents, will include techniques such as Latent Dirichlet Allocation (LDA), and [BERTopic](https://github.com/MaartenGr/BERTopic), after which topic representations will be fine-tuned by prompting an LLM to create short topic labels.

## Information Extraction Pipeline

Information Extraction is the higher-level task of extracting structured information from unstructured text and comprises a pipeline of subtasks, including Coreference Resolution, Named Entity Recognition (NER) and Linking, and Relation Extraction. The structured data can then be analysed, searched and visualised, and has a wide variety of applications across multiple domains. Constructing a knowledge graph to represent the extracted information about real-world entities and their relationships to each other could serve as a knowledge base for use in information retrieval, search and recommendation systems.

The pipeline will take the preprocessed text as input, resolve coreferences using a neural model, classify entities with spaCy/scispaCy and fine-tuned BERT-based NER and linking models, and use a seq2seq model based on the BART encoder-decoder transformer architecture, a GPT-style base LLM, and a supervised fine-tuned LLM to extract predefined semantic relations between named entities, and output subject-predicate-object triplets to build knowledge graphs. 


![Information extraction pipeline diagram](https://github.com/alisonmitchell/Biomedical-Knowledge-Graph/blob/main/images/information_extraction_pipeline.png?raw=true)


This hybrid approach to information extraction includes traditional rule-based, statistical machine learning, and neural network-based methods which require specialised pipelines for each task; generalist pre-trained language models (PLMs) leveraging the idea of transfer learning and supervised fine-tuning on domain/task-specific data; and LLMs which revolutionalised the NLP landscape by building on the transformer architecture to unprecedented levels. Models trained on vast datasets, and utilising billions of parameters to define behaviour, were able to understand and analyse natural language and generate human-like responses. 

An information extraction pipeline could be performed entirely using LLMs, and a single model with generalisation capabilities could solve a range of NLP tasks without fine-tuning. However, challenges and limitations persist as LLMs generate text based on probabilities. They have a knowledge cutoff date, and their training data does not include external domain-specific data sources, often resulting in hallucinations.


### Coreference Resolution

Coreference Resolution is the advanced preprocessing task of finding and establishing a relation between all linguistic expressions, or entity mentions, in a text that refer to the same real-world entity. Here it is represented as the task of replacing pronouns with referenced entities, and the [fastcoref](https://github.com/shon-otmazgin/fastcoref) neural model will be used to predict whether spans are coreferent.


### Named Entity Recognition

Named Entity Recognition (NER) models are trained to identify and classify specific entities in text into predefined categories. scispaCy NER models trained on biomedical corpora will be used to extract and label entities, and the EntityLinker component to query supported knowledge bases and match entities to entries. End-to-end models such as BioBERT-based [BERN2](https://github.com/dmis-lab/BERN2), and the more computationally efficient [KAZU](https://github.com/AstraZeneca/KAZU), perform NER and Linking in a single step. 


![Named entities and labels](https://github.com/alisonmitchell/Biomedical-Knowledge-Graph/blob/main/images/named_entities_and_labels.png?raw=true)


### Relation Extraction

Relation Extraction is the final step in the information extraction pipeline and is the process of extracting relationships between entities in the text. [REBEL](https://github.com/Babelscape/rebel) is a seq2seq model which performs end-to-end entity and relation extraction for more than 200 different relation types using a Hugging Face pipeline, as a spaCy component, or directly with the transformers library and generate() function. REBEL will also be used to extract triplets for indexing, storage and retrieval by [LlamaIndex](https://github.com/run-llama/llama_index) along with a pre-trained Sentence Transformers embedding model, and an LLM to respond to prompts and queries, and for writing natural language responses. During queries, keywords are extracted from the query text and used to find triplets containing the same subject, and their associated text chunks.

The use of LLMs for knowledge graph construction, and the leveraging of knowledge graphs to enhance LLMs, will be demonstrated using [LangChain](https://github.com/langchain-ai), an open-source orchestration framework for developing context-aware, reasoning applications powered by language models.


[Groq](https://console.groq.com/playground)'s fast AI inference API for supported LLMs, with few-shot prompting to extract triples, will demonstrate the generalisation capabilities of LLMs over domain and task-specific NLP models. Finally, [Unsloth](https://github.com/unslothai/unsloth) will be used to simplify and accelerate the supervised fine-tuning process for a supported LLM using a gold standard annotated dataset with labelled training examples and expected outputs.


## Knowledge Graph

A knowledge graph is a graphic representation of a network of real-world entities and the relationships between them. The term 'Knowledge Graph' was officially coined by [Google](https://blog.google/products/search/introducing-knowledge-graph-things-not/) in 2012, although conceptually it dates back to the 1950s and the advent of the digital age. The adoption of knowledge graphs (KGs) in NLP has become increasingly popular in recent years with frameworks to [unify LLMs and KGs](https://arxiv.org/pdf/2306.08302) and simultaneously leverage the advantages of KG-enhanced LLMs, LLM-augmented KGs, and Synergised LLMs + KGs. Indeed, [Gartner](https://www.gartner.com/en/newsroom/press-releases/2021-03-16-gartner-identifies-top-10-data-and-analytics-technologies-trends-for-2021) predicted that knowledge graphs would be a part of 80% of data and analytics innovations by 2025, up from 10% in 2021, and that [composite AI](https://www.gartner.com/en/articles/hype-cycle-for-artificial-intelligence) represents the next phase in AI evolution by combining AI methodologies — such as machine learning, natural language processing and knowledge graphs — to create more adaptable and scalable solutions.

Here, [NetworkX](https://github.com/networkx/networkx) and [Pyvis](https://github.com/WestHealth/pyvis) will be used to visualise the output from the information extraction process as interactive network graphs of colour-coded nodes with properties comprising biomedical entities, entity types, and entity IDs linking to target knowledge bases, and edges representing the semantic relationships between them.


![Pyvis knowledge graph with node tooltip for alveolar type ii cells](https://github.com/alisonmitchell/Biomedical-Knowledge-Graph/blob/main/images/pyvis_knowledge_graph_alveolar_type_ii_cells.png?raw=true)


## Data sources

* [Europe PMC](https://europepmc.org/)

* [arXiv](https://arxiv.org/)


## Python libraries

* [arxiv.py](https://github.com/lukasschwab/arxiv.py)

* [GROBID](https://github.com/kermitt2/grobid)

* [scipdf_parser](https://github.com/titipata/scipdf_parser)

* [Europe PMC Articles RESTful API](https://europepmc.org/RestfulWebService)

* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

* [pdfminer.six](https://github.com/pdfminer/pdfminer.six)

* [pdfplumber](https://github.com/jsvine/pdfplumber)

* [spaCy](https://github.com/explosion/spaCy)

* [scispaCy](https://github.com/allenai/scispacy)

* [Textstat](https://github.com/textstat/textstat)

* [Plotly](https://plotly.com/python/)

* [kneed](https://github.com/arvkevi/kneed)

* [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick)

* [UMAP](https://github.com/lmcinnes/umap)

* [Gensim](https://github.com/piskvorky/gensim)

* [embedding-explorer](https://github.com/centre-for-humanities-computing/embedding-explorer)

* [Embetter](https://github.com/koaning/embetter)

* [glove-python](https://github.com/maciejkula/glove-python)

* [glovpy](https://github.com/centre-for-humanities-computing/glovpy)

* [openai-python](https://github.com/openai/openai-python)

* [BERTopic](https://github.com/MaartenGr/BERTopic)

* [fastcoref](https://github.com/shon-otmazgin/fastcoref)

* [BERN2](https://github.com/dmis-lab/BERN2)

* [KAZU](https://github.com/AstraZeneca/KAZU)

* [REBEL](https://github.com/Babelscape/rebel)

* [Groq](https://groq.com/)

* [Hugging Face](https://huggingface.co/docs/transformers/en/index)

* [LlamaIndex](https://github.com/run-llama/llama_index)

* [MELODI Presto](https://melodi-presto.mrcieu.ac.uk/)

* [Unsloth](https://github.com/unslothai/unsloth)

* [NetworkX](https://github.com/networkx/networkx)

* [Pyvis](https://github.com/WestHealth/pyvis)

