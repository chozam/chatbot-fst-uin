from rag import RetrieavalAugmentedGeneration
import os

rag_orches = RetrieavalAugmentedGeneration()

for i in os.listdir('./knowledge_doc'):
    rag_orches.load_to_supabase(f'./knowledge_doc/{i}')\
    
print("Completed")