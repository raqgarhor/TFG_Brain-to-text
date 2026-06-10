package com.tfg.brain_to_text_web.service;

import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.repository.TrialRepository;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class TrialService {

    private final TrialRepository trialRepository;

    public TrialService(TrialRepository trialRepository) {
        this.trialRepository = trialRepository;
    }

    @Transactional(readOnly = true)
    public List<String> findSplits() {
        return trialRepository.findDistinctSplits();
    }

    @Transactional(readOnly = true)
    public Page<Trial> findTrialsPage(String split, int page, int size) {
        Sort sort = Sort.by("sessionName").ascending()
                .and(Sort.by("split").ascending())
                .and(Sort.by("trialKey").ascending());
        PageRequest pageRequest = PageRequest.of(Math.max(page, 0), size, sort);

        if (split == null || split.isBlank() || split.equals("all")) {
            return trialRepository.findAll(pageRequest);
        }
        return trialRepository.findBySplit(split, pageRequest);
    }

    @Transactional(readOnly = true)
    public Trial findTrialWithPredictions(Long id) {
        return trialRepository.findWithPredictionsById(id)
                .orElseThrow(() -> new IllegalArgumentException("Ensayo no encontrado: " + id));
    }

    @Transactional(readOnly = true)
    public Trial findTrial(Long id) {
        return trialRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Ensayo no encontrado: " + id));
    }

    @Transactional
    public Trial createTrial(
            String sessionName,
            String split,
            String trialKey,
            String realSentence,
            String realPhonemes,
            String inputShape,
            String logitsShape,
            String signalImagePath,
            String notes
    ) {
        Trial trial = new Trial();
        trial.setSessionName(sessionName);
        trial.setSplit(split);
        trial.setTrialKey(trialKey);
        trial.setRealSentence(emptyToNull(realSentence));
        trial.setRealPhonemes(emptyToNull(realPhonemes));
        trial.setInputShape(emptyToNull(inputShape));
        trial.setLogitsShape(emptyToNull(logitsShape));
        trial.setSignalImagePath(emptyToNull(signalImagePath));
        trial.setNotes(emptyToNull(notes));
        return trialRepository.save(trial);
    }

    @Transactional
    public Trial updateTrial(
            Long id,
            String sessionName,
            String split,
            String trialKey,
            String realSentence,
            String realPhonemes,
            String inputShape,
            String logitsShape,
            String signalImagePath,
            String notes
    ) {
        Trial trial = findTrial(id);
        trial.setSessionName(sessionName);
        trial.setSplit(split);
        trial.setTrialKey(trialKey);
        trial.setRealSentence(emptyToNull(realSentence));
        trial.setRealPhonemes(emptyToNull(realPhonemes));
        trial.setInputShape(emptyToNull(inputShape));
        trial.setLogitsShape(emptyToNull(logitsShape));
        trial.setSignalImagePath(emptyToNull(signalImagePath));
        trial.setNotes(emptyToNull(notes));
        return trialRepository.save(trial);
    }

    @Transactional
    public void deleteTrial(Long id) {
        trialRepository.deleteById(id);
    }

    @Transactional(readOnly = true)
    public long countTrials() {
        return trialRepository.count();
    }

    private String emptyToNull(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.trim();
    }
}
