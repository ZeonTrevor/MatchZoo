cd ../../

currpath=`pwd`
# train the model
python matchzoo/main.py --phase train --model_file ${currpath}/examples/robust04/config/duet_ranking.config


# predict with the model

python matchzoo/main.py --phase predict --model_file ${currpath}/examples/robust04/config/duet_ranking.config
