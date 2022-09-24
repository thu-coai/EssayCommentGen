
if [ $1 == 0 ]; then
    echo 'generate kws'
    task=test_notitledup
    cuda=0
    k=3164 # 需要替换为最好checkpoint的step数
    echo $k
    for mode in all
    do
    echo $mode;
    python3 -u ./gen.py cg_base_notitledup_tfidfkws $k $cuda $task $mode;
done

elif [ $1 == 1 ]; then
    echo 'generate comments using kws'
    task=test_notitledup_generated_tfidfkws_corrected
    cuda=0
    k=5650 # 需要替换为最好checkpoint的step数
    echo $k 
    for mode in all
    do
    echo $mode;
    python3 -u ./gen.py cg_base_notitledup_tfidfkws_to_comment_noised $k $cuda $task $mode;
    done
