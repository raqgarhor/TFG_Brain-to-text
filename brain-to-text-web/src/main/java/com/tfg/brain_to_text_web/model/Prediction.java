package com.tfg.brain_to_text_web.model;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "predictions")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "trial_id", nullable = false)
    private Trial trial;

    @Column(name = "model_name", nullable = false)
    private String modelName;

    @Column(name = "model_label", nullable = false)
    private String modelLabel;

    @Column(name = "predicted_phonemes")
    private String predictedPhonemes;

    @Column(name = "predicted_text")
    private String predictedText;

    @Column(name = "partial_text")
    private String partialText;

    @Column(name = "per_value")
    private BigDecimal perValue;

    @Column(name = "wer_value")
    private BigDecimal werValue;

    @Column(name = "checkpoint_per")
    private BigDecimal checkpointPer;

    private String notes;

    @Column(name = "created_at", insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    @Column(name = "updated_at", insertable = false, updatable = false)
    private OffsetDateTime updatedAt;

    @OneToMany(mappedBy = "prediction", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @OrderBy("rank asc")
    private List<Candidate> candidates = new ArrayList<>();

    public Long getId() {
        return id;
    }

    public Trial getTrial() {
        return trial;
    }

    public void setTrial(Trial trial) {
        this.trial = trial;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getModelLabel() {
        return modelLabel;
    }

    public void setModelLabel(String modelLabel) {
        this.modelLabel = modelLabel;
    }

    public String getPredictedPhonemes() {
        return predictedPhonemes;
    }

    public void setPredictedPhonemes(String predictedPhonemes) {
        this.predictedPhonemes = predictedPhonemes;
    }

    public String getPredictedText() {
        return predictedText;
    }

    public void setPredictedText(String predictedText) {
        this.predictedText = predictedText;
    }

    public String getPartialText() {
        return partialText;
    }

    public void setPartialText(String partialText) {
        this.partialText = partialText;
    }

    public BigDecimal getPerValue() {
        return perValue;
    }

    public void setPerValue(BigDecimal perValue) {
        this.perValue = perValue;
    }

    public BigDecimal getWerValue() {
        return werValue;
    }

    public void setWerValue(BigDecimal werValue) {
        this.werValue = werValue;
    }

    public BigDecimal getCheckpointPer() {
        return checkpointPer;
    }

    public void setCheckpointPer(BigDecimal checkpointPer) {
        this.checkpointPer = checkpointPer;
    }

    public String getNotes() {
        return notes;
    }

    public void setNotes(String notes) {
        this.notes = notes;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public OffsetDateTime getUpdatedAt() {
        return updatedAt;
    }

    public List<Candidate> getCandidates() {
        return candidates;
    }

    public void setCandidates(List<Candidate> candidates) {
        this.candidates = candidates;
    }
}
