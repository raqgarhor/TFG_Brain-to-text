package com.tfg.brain_to_text_web.service;

import com.tfg.brain_to_text_web.model.AppUser;
import com.tfg.brain_to_text_web.model.Candidate;
import com.tfg.brain_to_text_web.model.Prediction;
import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.repository.AppUserRepository;
import com.tfg.brain_to_text_web.repository.CandidateRepository;
import com.tfg.brain_to_text_web.repository.PredictionRepository;
import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AdminService {

    private final AppUserRepository appUserRepository;
    private final PredictionRepository predictionRepository;
    private final CandidateRepository candidateRepository;
    private final TrialService trialService;

    public AdminService(
            AppUserRepository appUserRepository,
            PredictionRepository predictionRepository,
            CandidateRepository candidateRepository,
            TrialService trialService
    ) {
        this.appUserRepository = appUserRepository;
        this.predictionRepository = predictionRepository;
        this.candidateRepository = candidateRepository;
        this.trialService = trialService;
    }

    @Transactional(readOnly = true)
    public Optional<AppUser> authenticateAdmin(String username, String password) {
        if (username == null || password == null || username.isBlank() || password.isBlank()) {
            return Optional.empty();
        }
        return appUserRepository.findValidAdmin(username.trim(), password);
    }

    @Transactional(readOnly = true)
    public AdminStats getStats() {
        return new AdminStats(
                trialService.countTrials(),
                predictionRepository.count(),
                candidateRepository.count(),
                appUserRepository.count()
        );
    }

    @Transactional(readOnly = true)
    public List<Prediction> findPredictionsForTrial(Long trialId) {
        return predictionRepository.findByTrial_IdOrderByModelNameAsc(trialId);
    }

    @Transactional
    public Prediction createPrediction(
            Long trialId,
            String modelName,
            String modelLabel,
            String predictedPhonemes,
            String predictedText,
            String partialText,
            String perValue,
            String werValue,
            String checkpointPer,
            String notes
    ) {
        Trial trial = trialService.findTrial(trialId);
        Prediction prediction = new Prediction();
        prediction.setTrial(trial);
        fillPrediction(
                prediction,
                modelName,
                modelLabel,
                predictedPhonemes,
                predictedText,
                partialText,
                perValue,
                werValue,
                checkpointPer,
                notes
        );
        return predictionRepository.save(prediction);
    }

    @Transactional
    public Prediction updatePrediction(
            Long predictionId,
            String modelName,
            String modelLabel,
            String predictedPhonemes,
            String predictedText,
            String partialText,
            String perValue,
            String werValue,
            String checkpointPer,
            String notes
    ) {
        Prediction prediction = predictionRepository.findById(predictionId)
                .orElseThrow(() -> new IllegalArgumentException("Predicción no encontrada: " + predictionId));
        fillPrediction(
                prediction,
                modelName,
                modelLabel,
                predictedPhonemes,
                predictedText,
                partialText,
                perValue,
                werValue,
                checkpointPer,
                notes
        );
        return predictionRepository.save(prediction);
    }

    @Transactional
    public void deletePrediction(Long predictionId) {
        predictionRepository.deleteById(predictionId);
    }

    @Transactional
    public Candidate createCandidate(Long predictionId, String rank, String candidateText) {
        Prediction prediction = predictionRepository.findById(predictionId)
                .orElseThrow(() -> new IllegalArgumentException("Predicción no encontrada: " + predictionId));
        Candidate candidate = new Candidate();
        candidate.setPrediction(prediction);
        fillCandidate(candidate, rank, candidateText);
        return candidateRepository.save(candidate);
    }

    @Transactional
    public Candidate updateCandidate(Long candidateId, String rank, String candidateText) {
        Candidate candidate = candidateRepository.findById(candidateId)
                .orElseThrow(() -> new IllegalArgumentException("Candidato no encontrado: " + candidateId));
        fillCandidate(candidate, rank, candidateText);
        return candidateRepository.save(candidate);
    }

    @Transactional
    public void deleteCandidate(Long candidateId) {
        candidateRepository.deleteById(candidateId);
    }

    private void fillPrediction(
            Prediction prediction,
            String modelName,
            String modelLabel,
            String predictedPhonemes,
            String predictedText,
            String partialText,
            String perValue,
            String werValue,
            String checkpointPer,
            String notes
    ) {
        prediction.setModelName(required(modelName));
        prediction.setModelLabel(required(modelLabel));
        prediction.setPredictedPhonemes(emptyToNull(predictedPhonemes));
        prediction.setPredictedText(emptyToNull(predictedText));
        prediction.setPartialText(emptyToNull(partialText));
        prediction.setPerValue(parseDecimal(perValue));
        prediction.setWerValue(parseDecimal(werValue));
        prediction.setCheckpointPer(parseDecimal(checkpointPer));
        prediction.setNotes(emptyToNull(notes));
    }

    private void fillCandidate(Candidate candidate, String rank, String candidateText) {
        candidate.setRank(Integer.parseInt(required(rank)));
        candidate.setCandidateText(required(candidateText));
    }

    private BigDecimal parseDecimal(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return new BigDecimal(value.trim().replace(",", "."));
    }

    private String emptyToNull(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.trim();
    }

    private String required(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Campo obligatorio vacío");
        }
        return value.trim();
    }

    public record AdminStats(
            long trialCount,
            long predictionCount,
            long candidateCount,
            long userCount
    ) {
    }
}
