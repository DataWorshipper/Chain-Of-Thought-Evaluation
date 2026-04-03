import os
import pandas as pd
from dotenv import load_dotenv
from load_dataset import dataset
from thinker.thinker_agent import thinker_graph
from executor.executor_agent import final_pipeline, NUM_EXECUTORS 

def run_pipeline(num_samples=5):
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: RUNNING THINKER AGENT (Samples: {num_samples})")
    print(f"{'='*60}")
    
    all_thinker_results = []
    samples = dataset["train"].select(range(num_samples))
    
    for i, example in enumerate(samples):
        initial_state = {
            "id": i + 1,
            "question": example["question"],
            "true_answer": example["final_answer"]
        }
        
       
        result = thinker_graph.invoke(initial_state)
        
      
        record = {**initial_state, **result}
        all_thinker_results.append(record)
        
        print(f"Processed Thinker {i+1}/{num_samples} | Is Correct: {result.get('is_correct')}")

    print(f"\n{'='*60}")
    print("PHASE 2: DATAFRAME & CSV FILTERING")
    print(f"{'='*60}")
    
    df = pd.DataFrame(all_thinker_results)
    
  
    df_correct = df[df["is_correct"] == 1].copy()
    
    csv_filename = "thinker_correct_only.csv"
    df_correct.to_csv(csv_filename, index=False)
    
    print(f"Total questions processed: {len(df)}")
    print(f"Successfully answered by Thinker: {len(df_correct)}")
    print(f"Saved correct responses to: {csv_filename}")

  
    if df_correct.empty:
        print("\nPipeline halted: No correct Thinker answers available to pass to Executors.")
        return

    print(f"\n{'='*60}")
    print(f"PHASE 3: EXECUTOR PIPELINE FAN-OUT")
    print(f"{'='*60}")
    

    filtered_tasks = []
    for _, row in df_correct.iterrows():
        filtered_tasks.append({
            "task_id": row["id"],
            "question": row["question"],
            "correct_ans": row["true_answer"], 
            "thinker_cot": row["thinker_cot"]
        })

   
    initial_executor_state = {"filtered_tasks": filtered_tasks, "executor_results": []}
    final_state = final_pipeline.invoke(initial_executor_state)
    results_data = final_state.get("executor_results", [])

    print(f"\n{'='*60}")
    print("PHASE 4: FINAL METRICS (PER EXECUTOR)")
    print(f"{'='*60}")
    
   
    metrics_by_executor = {
        i: {"Independent Success": 0, "CoT Rescued": 0, "Total Failure": 0, "total_tasks": 0} 
        for i in range(1, NUM_EXECUTORS + 1)
    }
    
   
    for res in results_data:
        e_id = res.get("executor_id")
        status = res.get("status")
        if e_id and e_id in metrics_by_executor:
            metrics_by_executor[e_id][status] += 1
            metrics_by_executor[e_id]["total_tasks"] += 1
    
   
    for e_id in range(1, NUM_EXECUTORS + 1):
        stats = metrics_by_executor[e_id]
        ind_success = stats["Independent Success"]
        cot_rescued = stats["CoT Rescued"]
        total_fail = stats["Total Failure"]
        total = stats["total_tasks"]
        
        initial_failures = cot_rescued + total_fail
        reusability = (cot_rescued / initial_failures * 100) if initial_failures > 0 else 0.0
        
        print(f"\n[ EXECUTOR {e_id} ] - Processed {total} tasks")
        print(f"  ├─ Independent Successes : {ind_success}")
        print(f"  ├─ CoT Rescued           : {cot_rescued}")
        print(f"  ├─ Total Failures        : {total_fail}")
        
        if initial_failures > 0:
            print(f"  └─ Reusability Score     : {reusability:.2f}%")
        else:
            print(f"  └─ Reusability Score     : N/A (No initial failures)")

if __name__ == "__main__":
    load_dotenv()
    
   
    run_pipeline(num_samples=5)