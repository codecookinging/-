python finetune.py \
	--learning_rate "0.00001" \
	--num_epochs 25 \
	--multi_scale "225,256" \
    --train_layers "fc8,fc7,fc6,conv5_3,conv5_2,con5_1,conv4_3,conv4_2,conv4_1,conv3_3,conv3_2,conv3_1,conv2_2,conv2_1,conv1_2,conv1_1"