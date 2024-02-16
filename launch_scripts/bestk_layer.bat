
set target_domain=cartoon
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_first/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=first



@REM

set target_domain=sketch
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_all/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=all


set target_domain=photo
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_all/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=all

set target_domain=cartoon
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_all/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=all


@REM

set target_domain=sketch
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate


set target_domain=photo
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate

set target_domain=cartoon
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate



@REM

set target_domain=sketch
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate_3/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate_3


set target_domain=photo
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate_3/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate_3

set target_domain=cartoon
python main.py ^
--experiment=topKvalue ^
--experiment_name=topKvalue_20_alternate_3/%target_domain%/ ^
--dataset_args="{'root': 'C:\\\\Users\\giobo\\Documents\\Magistrale_2anno\\AML\\project_shared\\Activation_shaping_for_domain_adaptation\\data\\PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%','K' : 0.2}" ^
--batch_size=128 ^
--num_workers=2 ^
--grad_accum_steps=2 ^
--layer=alternate_3