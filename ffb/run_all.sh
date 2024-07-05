#!/bin/bash

# adult dataset
for model in adv_gr diffdp erm hsic laftr pr; do
    for sensitive_attr in race sex; do
        python ffb_tabular_${model}.py --dataset adult --target_attr income --sensitive_attr ${sensitive_attr} &
    done
done
wait

# german dataset, needs smaller batch size, because of small dataset
for model in adv_gr diffdp erm hsic laftr pr; do
    for sensitive_attr in sex age; do
        python ffb_tabular_${model}.py --dataset german --target_attr credit --sensitive_attr ${sensitive_attr} --batch_size 128 & 
    done
done
wait

# bank_marketing dataset
for model in adv_gr diffdp erm hsic laftr pr; do
    for sensitive_attr in age; do
        python ffb_tabular_${model}.py --dataset bank_marketing --target_attr y --sensitive_attr ${sensitive_attr} &
    done
done
wait

# compas dataset
for model in adv_gr diffdp erm hsic laftr pr; do
    for sensitive_attr in race sex; do
        python ffb_tabular_${model}.py --dataset compas --target_attr two_year_recid --sensitive_attr ${sensitive_attr} &
    done
done
wait