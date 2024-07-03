#!/bin/bash

# adult dataset
for model in adv_gr diffdp erm hsic laftr pr; do
    for sensitive_attr in race sex; do
        python ffb_tabular_${model}.py --dataset adult --target_attr income --sensitive_attr ${sensitive_attr}
    done
done



