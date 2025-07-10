#finetune_r2plus1d_w_knowledge_ucfCMN_ablationNoOrig.sh
cd C:/Futures/Knowledge-Prompting-for-FSAR/ && CUDA_VISIBLE_DEVICES=0 python finetune_metatrain_w_knowledge.py --video_path D:/Data/Frame_Plat_fps_5  \
--train_list_path C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/data_spilts/meta_train.txt \
--val_video_path C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/frame_classes_divided_plat \
--val_list_path C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/data_spilts/meta_val.txt \
--dataset ucf101 \
--n_classes 14 \
--n_finetune_classes 9 \
--knowledge_model dwconv_fc \
--model_depth 34 \
--shot 1 \
--train_way 5 \
--batch_size 4 \
--n_threads 0 \
--checkpoint 1 \
--val_every 10 \
--train_crop random \
--n_samples_for_each_video 8 \
--n_val_samples 10 \
--weight_decay 0.001 \
--layer_lr 0.005 0.005 0.005 0.005 0.005 0.1 \
--ft_begin_index 0 \
--result_path C:/Futures/Knowledge-Prompting-for-FSAR/results_newlr/plat_5_1shot/train_knowbase_1e-4_finegym_ins \
--CLIP_visual_fea_reg "C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/clip_related/VitOutput_plat/**/*" \
--proposals_fea_pth C:/Futures/Knowledge-Prompting-for-FSAR/data_plat/clip_related/combined_proposals_fea.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 16 --is_w_knowledge --is_amp --dropout_w_knowledge 0.9 \
--this_launch_script $0 --print_freq 200 --sample_mode sparse --n_epochs 50 --temporal_modeling TSM1 --sample_duration 16

# batch size 32     
#opt.n_finetune_classes 是训练集的类别数，opt.n_classes是总的类别数
