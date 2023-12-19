target_domain=${1}

python main.py \
--experiment=baseline \
--experiment_name=baseline/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=2\
--grad_accum_steps=2