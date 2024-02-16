set target_domain=photo
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_02middle/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%', 'K' : 0.2 }" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=middle

set target_domain=cartoon
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_02middle/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%', 'K' : 0.2 }" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=middle



set target_domain=sketch
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_02middle/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%', 'K' : 0.2 }" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=middle