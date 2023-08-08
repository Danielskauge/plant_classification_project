# Automated Plant Classification
## Brisbane Flora and Fauna Society

### 1. About Us
The Brisbane Flora and Fauna Society (BFFS) is a not-for-profit committed to the preservation, protection, and enhancement of Brisbane's diverse ecosystem. BFFS is composed of dedicated environmental enthusiasts, biologists, and volunteers who work tirelessly to safeguard the region's rich biodiversity, focusing specifically on the city's unique flora and fauna. Through education, conservation initiatives, community participation, and collaboration with government and non-government entities, BFFS is a frontline guardian of Brisbane's natural heritage, balancing urban development with ecological sustainability.

### 2. Project Motivation
We currently use trained experts to monitor the growth of different plant species in Brisbane parklands. These experts label and count the ‘healthy’ plants in the parks – these are plants that encourage biodiversity. The experts also identify and remove ‘weeds’ – plants that grow too quickly and kill other healthy plants. This existing approach works well, but is difficult to scale as training experts is expensive and time-consuming.

We want to trial a new approach where non-expert volunteers upload a photo from a park and an ML algorithm classifies the species of the plant. If the photo is classified as a healthy plant, we will use that to monitor the population of that specific species. If the photo is classified as a weed, the volunteer will mark that plant for future removal or weed-spraying.

### 3. Project Description
We want you to complete a study into the feasibility of using AI to classify plant species. We have collected data for 5 different healthy species and 5 different weed species.

#### Dataset:
- **labelled** – This folder contains example images from the 10 different species, sorted into individual species folders. We have used one of our trained experts to label this data. There are between 40-70 labelled images for each plant species. There are a total of 548 images.
- **unlabelled** – This folder contains approximately 430 extra images of plant species that have not been labelled by an expert. The majority of these images should belong to one of the provided 10 species, and there may be a small number that do not belong to any of the 10 species.

#### Held-Out Set:
We additionally have a held-out set of 300 images, with 30 images per plant species, that we will use to evaluate the AI method you design. We will not release these images to you, but we will test your final solution on this data to measure the reliability of your method.

### 4. Project Deliverables
#### Constraint 1: The model
- For classifying images, use the Pytorch implementation of a ResNet18 architecture initialised with the available pre-trained weights from training on ImageNet.
- You may change the final layer of this model to be suitable for this dataset with 10 classes.

#### Constraint 2: Data subsets
- You should create a training and validation subset from the data you have been provided. Each plant species must have at least 20 images in the validation subset. When presenting your results, you should report performance on your validation subset.

#### Tasks:
1. **Task 1:** What performance can you achieve with only the labelled data?
2. **Task 2:** How can you utilise the unlabelled data as well as the labelled data? How does this affect training and performance?
3. **Task 3:** Given your investigation, do you have any recommendations for us?
4. **Task 4:** Provide us with your recommended ML algorithm.

### 6. Project Deadline
We look forward to meeting with you in Week 7. We will reach out to you in the coming weeks to schedule a 10-minute meeting, where you can present your findings to one of our representatives.
