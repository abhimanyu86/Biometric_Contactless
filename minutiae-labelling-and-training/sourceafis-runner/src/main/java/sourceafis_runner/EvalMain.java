package sourceafis_runner;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * SourceAFIS match-score benchmark on the same 336-person contactless
 * fingerprint dataset (first_session, p1..p336, p1.bmp=gallery, p2..p6.bmp
 * =probes) already used for the MinutiaeNet baseline and the CNN/Siamese
 * embedding Phase A evaluation in siamese_embedding/. SourceAFIS's public
 * API doesn't expose minutiae coordinates (checked earlier, ruled out for
 * ground-truth labeling), but it does output a genuine match score, which
 * is exactly what's needed for a FAR/FRR/EER comparison against the other
 * matchers already benchmarked on this project.
 *
 * Usage: EvalMain <datasetRoot> <outCsv> <numPeople> <maxImpostorsPerProbe>
 * maxImpostorsPerProbe = -1 means match every probe against every other
 * person's gallery template (full pairwise, ~562K pairs at numPeople=336).
 */
public class EvalMain {
    public static void main(String[] args) throws Exception {
        String datasetRoot = args[0];
        String outCsv = args[1];
        int numPeople = Integer.parseInt(args[2]);
        int maxImpostorsPerProbe = Integer.parseInt(args[3]);

        FingerprintImageOptions options = new FingerprintImageOptions().dpi(500);

        List<Integer> validIds = new ArrayList<>();
        FingerprintTemplate[] gallery = new FingerprintTemplate[numPeople + 1];
        List<FingerprintTemplate>[] probes = new List[numPeople + 1];

        long loadStart = System.currentTimeMillis();
        for (int i = 1; i <= numPeople; i++) {
            Path galleryPath = Paths.get(datasetRoot, "p" + i, "p1.bmp");
            if (!Files.exists(galleryPath)) {
                System.out.println("[load] WARNING missing gallery for p" + i + ", skipping");
                continue;
            }
            byte[] galleryBytes = Files.readAllBytes(galleryPath);
            gallery[i] = new FingerprintTemplate(new FingerprintImage(galleryBytes, options));

            List<FingerprintTemplate> probeList = new ArrayList<>();
            for (int j = 2; j <= 6; j++) {
                Path probePath = Paths.get(datasetRoot, "p" + i, "p" + j + ".bmp");
                if (Files.exists(probePath)) {
                    byte[] b = Files.readAllBytes(probePath);
                    probeList.add(new FingerprintTemplate(new FingerprintImage(b, options)));
                }
            }
            probes[i] = probeList;
            validIds.add(i);

            if (i % 20 == 0 || i == numPeople) {
                System.out.println("[load] " + i + "/" + numPeople + " people loaded ("
                        + (System.currentTimeMillis() - loadStart) + " ms elapsed)");
            }
        }
        System.out.println("[load] done: " + validIds.size() + " people with valid templates, "
                + (System.currentTimeMillis() - loadStart) + " ms");

        Random rng = new Random(42);
        BufferedWriter w = new BufferedWriter(new FileWriter(outCsv));
        w.write("pair_type,person,probe_index,other_person,score\n");

        long matchStart = System.currentTimeMillis();
        long genuineCount = 0, impostorCount = 0;
        int probesDone = 0;
        int totalProbes = 0;
        for (int pid : validIds) totalProbes += probes[pid].size();

        for (int pid : validIds) {
            List<FingerprintTemplate> probeList = probes[pid];
            for (int pIdx = 0; pIdx < probeList.size(); pIdx++) {
                FingerprintTemplate probeTemplate = probeList.get(pIdx);
                FingerprintMatcher matcher = new FingerprintMatcher(probeTemplate);

                double genuineScore = matcher.match(gallery[pid]);
                w.write("genuine," + pid + "," + pIdx + ",," + genuineScore + "\n");
                genuineCount++;

                List<Integer> others;
                if (maxImpostorsPerProbe < 0) {
                    others = validIds;
                } else {
                    others = new ArrayList<>(validIds);
                    java.util.Collections.shuffle(others, rng);
                    others = others.subList(0, Math.min(maxImpostorsPerProbe, others.size()));
                }
                for (int otherId : others) {
                    if (otherId == pid) continue;
                    double impostorScore = matcher.match(gallery[otherId]);
                    w.write("impostor," + pid + "," + pIdx + "," + otherId + "," + impostorScore + "\n");
                    impostorCount++;
                }

                probesDone++;
                if (probesDone % 50 == 0 || probesDone == totalProbes) {
                    long elapsed = System.currentTimeMillis() - matchStart;
                    System.out.println("[match] " + probesDone + "/" + totalProbes + " probes done, "
                            + genuineCount + " genuine + " + impostorCount + " impostor scores so far, "
                            + elapsed + " ms elapsed");
                }
            }
        }
        w.close();
        System.out.println("[match] FINISHED: " + genuineCount + " genuine, " + impostorCount
                + " impostor pairs written to " + outCsv + " in "
                + (System.currentTimeMillis() - matchStart) + " ms");
    }
}
