#test_knowledgeModel_dwconv_fc_ucfCMN_ablationNoOrig.sh

cd C:/Futures/Knowledge-Prompting-for-FSAR/ && CUDA_VISIBLE_DEVICES=0 python tsl_fsv_w_knowledge_plat.py --test_video_path  D:/Data/Frame_Plat_fps_5 \
--manual_seed 10 \
--test_list_path C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/data_spilts/meta_test.txt \
--dataset ucf101 \
--train_crop random \
--n_samples_for_each_video 10 \
--knowledge_model dwconv_fc \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--clip_model_depth 34 \
--n_threads 0 \
--result_path C:/Futures/Knowledge-Prompting-for-FSAR/results_newlr/plat_3_1shot/test_knowbase_1e-4_finegym_ins \
--test_way 3 \
--shot 1 \
--query 1 \
--resume_path C:/Futures/Knowledge-Prompting-for-FSAR/results_newlr/plat_3_1shot/train_knowbase_1e-4_finegym_ins/save_49.pth \
--emb_dim 491 \
--batch_size 64 \
--lr 0.05 \
--nepoch 10 \
--CLIP_visual_fea_reg "C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/clip_related/VitOutput_plat/**/*" \
--proposals_fea_pth C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/clip_related/combined_proposals_fea.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 5 --is_w_knowledge \
--is_amp --this_launch_script $0 --ablation_removeOrig --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 #--grad_enabled_in_embeddin #-ablation_onlyCLIPvisfea #--CLIP_visual_fea_preload
# --emb_dim 448 \
# --emb_dim 477 \
# --emb_dim 450 \
# --emb_dim 514 \

