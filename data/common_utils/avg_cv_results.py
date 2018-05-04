from __future__ import print_function
import subprocess

if __name__ == "__main__":
    basedir = "../robust04/cv_splits/"
    trec_eval = basedir + "trec_eval.9.0/trec_eval"
    qrels_file = basedir + "qrels.robust2004.txt"

    predict_baseline_files = [basedir + "predict.test.1.ql.txt",
                               basedir + "predict.test.2.ql.txt",
                               basedir + "predict.test.3.ql.txt",
                               basedir + "predict.test.4.ql.txt",
                               basedir + "predict.test.5.ql.txt"]
    predict_files = [basedir + "predict.test.1.iter400.bs20.steps1000.dropout0.lr0_001.tv.txt",
                     basedir + "predict.test.2.iter100.bs20.steps1000.dropout0.lr0_001.tv.txt",
                     basedir + "predict.test.3.iter100.bs20.steps1000.dropout0.lr0_001.tv.txt",
                     basedir + "predict.test.4.iter100.bs20.steps1000.dropout0.lr0_001.tv.txt",
                     basedir + "predict.test.5.iter100.bs20.steps1000.dropout0.lr0_001.tv.txt"]

    evaluation_metrics_file = basedir + "eval_scores.txt"
    fwrite = open(evaluation_metrics_file, 'w')

    #args = (trec_eval, "-m", "map", qrels_file, predict_file)
    metrics_avg = {}
    for id, baseline_file in enumerate(predict_baseline_files):
        print("\nmetrics for ql:%s and predict:%s" % (predict_baseline_files[id], predict_files[id]), end="\n")
        fwrite.write("\nmetrics for ql:%s and predict:%s\n" % (predict_baseline_files[id], predict_files[id]))

        args = trec_eval + " -m map -m recip_rank -m P.10,20 -m ndcg_cut.10,20 " + qrels_file + " " + predict_baseline_files[id]
        args.split()
        metrics = {}
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in p.stdout.readlines():
            line = line.split()
            if line[0] not in metrics:
                metrics[line[0]] = {}
            if line[0] not in metrics_avg:
                metrics_avg[line[0]] = {}
                metrics_avg[line[0]]['baseline'] = 0.0
                metrics_avg[line[0]]['drmm'] = 0.0

            metrics[line[0]]['baseline'] = line[2]
            metrics_avg[line[0]]['baseline'] += float(line[2])
            #print("%s %s" % (line[0],line[2]), end='\n')


        args = trec_eval + " -m map -m recip_rank -m P.10,20 -m ndcg_cut.10,20 " + qrels_file + " " + predict_files[id]
        args.split()
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in p.stdout.readlines():
            line = line.split()
            if line[0] not in metrics:
                metrics[line[0]] = {}
            metrics[line[0]]['drmm'] = line[2]
            metrics_avg[line[0]]['drmm'] += float(line[2])
            #print("%s %s" % (line[0],line[2]), end='\n')


        #col_width = max(len(metrics[metric][run]) for metric in metrics for run in metrics[metric]) + 2
        print("QL_baseline\tdrmm\tmetric")
        fwrite.write("QL_baseline\tdrmm\tmetric\n")
        for metric in sorted(metrics.iterkeys()):
            #print("".join(metrics[metric][run].ljust(col_width) for run in metrics[metric]), end="")
            #print(metric)
            print("%s\t\t%s\t%s" % (metrics[metric]['baseline'], metrics[metric]['drmm'], metric), end="\n")
            fwrite.write("%s\t\t%s\t%s\n" % (metrics[metric]['baseline'], metrics[metric]['drmm'], metric))
        retval = p.wait()

    print('\nCross Validation averages')
    fwrite.write('\nCross Validation averages\n')
    print("QL_baseline\tdrmm\t\tmetric")
    fwrite.write('QL_baseline\tdrmm\t\tmetric\n')
    for metric in sorted(metrics_avg.iterkeys()):
        print("%f\t%f\t%s" % ((metrics_avg[metric]['baseline']/5), (metrics_avg[metric]['drmm']/5), metric), end="\n")
        fwrite.write("%f\t%f\t%s\n" % ((metrics_avg[metric]['baseline']/5), (metrics_avg[metric]['drmm']/5), metric))

    fwrite.close()