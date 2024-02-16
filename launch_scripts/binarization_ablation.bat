set target_domain=sketch
python main.py ^
--experiment=binarization_ablation ^
--experiment_name=binarization_ablation_alternate_3/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%'}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate_3

set target_domain=photo
python main.py ^
--experiment=binarization_ablation ^
--experiment_name=binarization_ablation_alternate_3/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%'}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate_3

set target_domain=cartoon
python main.py ^
--experiment=binarization_ablation ^
--experiment_name=binarization_ablation_all/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%'}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=all
