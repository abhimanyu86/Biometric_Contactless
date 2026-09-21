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

/**
 * SourceAFIS match-score benchmark on the 30-person REAL phone-camera set
 * (sourceafis_input_30/, cropped via crop_30_for_sourceafis.py from
 * image_db/test_db -- the same 30-person real dataset already used for the
 * MinutiaeNet / ResNet18 / ResNet50 / ResNet50-finetuned comparisons in
 * siamese_embedding/). Directory layout: Person{id}/gallery.png,
 * probe1.png, probe2.png -- unlike the 336-person academic set, person
 * count and probe count per person are irregular (a YOLO crop can fail).
 *
 * Usage: EvalMain30 <inputRoot> <outCsv> <numPeople>
 */
public class EvalMain30 {
    public static void main(String[] args) throws Exception {
        String inputRoot = args[0];
        String outCsv = args[1];
        int numPeople = Integer.parseInt(args[2]);

        FingerprintImageOptions options = new FingerprintImageOptions().dpi(500);

        List<Integer> validIds = new ArrayList<>();
        FingerprintTemplate[] gallery = new FingerprintTemplate[numPeople + 1];
        List<FingerprintTemplate>[] probes = new List[numPeople + 1];

        for (int i = 1; i <= numPeople; i++) {
            Path personDir = Paths.get(inputRoot, "Person" + i);
            Path galleryPath = personDir.resolve("gallery.png");
            if (!Files.exists(galleryPath)) {
                System.out.println("[load] WARNING missing gallery for Person" + i + ", skipping");
                continue;
            }
            gallery[i] = new FingerprintTemplate(new FingerprintImage(Files.readAllBytes(galleryPath), options));

            List<FingerprintTemplate> probeList = new ArrayList<>();
            for (int j = 1; j <= 2; j++) {
                Path probePath = personDir.resolve("probe" + j + ".png");
                if (Files.exists(probePath)) {
                    probeList.add(new FingerprintTemplate(new FingerprintImage(Files.readAllBytes(probePath), options)));
                }
            }
            if (probeList.isEmpty()) {
                System.out.println("[load] WARNING no probes for Person" + i + ", skipping");
                continue;
            }
            probes[i] = probeList;
            validIds.add(i);
            System.out.println("[load] Person" + i + ": " + probeList.size() + " probe(s) loaded");
        }
        System.out.println("[load] done: " + validIds.size() + "/" + numPeople + " people usable");

        BufferedWriter w = new BufferedWriter(new FileWriter(outCsv));
        w.write("pair_type,person,probe_index,other_person,score\n");

        long genuineCount = 0, impostorCount = 0;
        for (int pid : validIds) {
            List<FingerprintTemplate> probeList = probes[pid];
            for (int pIdx = 0; pIdx < probeList.size(); pIdx++) {
                FingerprintTemplate probeTemplate = probeList.get(pIdx);
                FingerprintMatcher matcher = new FingerprintMatcher(probeTemplate);

                double genuineScore = matcher.match(gallery[pid]);
                w.write("genuine," + pid + "," + pIdx + ",," + genuineScore + "\n");
                genuineCount++;

                for (int otherId : validIds) {
                    if (otherId == pid) continue;
                    double impostorScore = matcher.match(gallery[otherId]);
                    w.write("impostor," + pid + "," + pIdx + "," + otherId + "," + impostorScore + "\n");
                    impostorCount++;
                }
            }
        }
        w.close();
        System.out.println("[match] FINISHED: " + genuineCount + " genuine, " + impostorCount
                + " impostor pairs written to " + outCsv);
    }
}
