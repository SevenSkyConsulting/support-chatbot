The objective here is to fine-tune a pre-trained chatbot LLM to respond to customer support questions with more precision.
First we need gather a huge dataset of customer support interactions history Q/A which in this case we will be using the "bitext chatbot dataset" from Huggingface.
Then we choose an chatbot LLM suitable for fine-tuning; one that is open-source and relatively small. for our use case here the small version Google's Flan T5 which is an already fine-tuned model of T5 for mixed tasks, is a good candidate.
Pre-processing and tokenizing the data is the next step and we move on to setting up Lora configuration and training arguments. 
Now everything is ready to use the Seq2Seq Trainer class to fine-tune our model.
Bear in mind that fine-tuning a model is quite costly in terms of GPU resources and also time-consuming even with the new Nvidia RTX GPUs. However you can slice your dataset for quicker results and educational purposes.
Saving the trained model on Huggingface for easier deployment and developing a DJango/Flask User interface would be our next steps to put this support chatbot to production.
