package com.tfg.brain_to_text_web.model;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "trials")
public class Trial {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "session_name", nullable = false)
    private String sessionName;

    @Column(nullable = false)
    private String split;

    @Column(name = "trial_key", nullable = false)
    private String trialKey;

    @Column(name = "real_sentence")
    private String realSentence;

    @Column(name = "real_phonemes")
    private String realPhonemes;

    @Column(name = "input_shape")
    private String inputShape;

    @Column(name = "logits_shape")
    private String logitsShape;

    private String notes;

    @Column(name = "created_at", insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    @Column(name = "updated_at", insertable = false, updatable = false)
    private OffsetDateTime updatedAt;

    @OneToMany(mappedBy = "trial", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @OrderBy("modelName asc")
    private List<Prediction> predictions = new ArrayList<>();

    public Long getId() {
        return id;
    }

    public String getSessionName() {
        return sessionName;
    }

    public void setSessionName(String sessionName) {
        this.sessionName = sessionName;
    }

    public String getSplit() {
        return split;
    }

    public void setSplit(String split) {
        this.split = split;
    }

    public String getTrialKey() {
        return trialKey;
    }

    public void setTrialKey(String trialKey) {
        this.trialKey = trialKey;
    }

    public String getRealSentence() {
        return realSentence;
    }

    public void setRealSentence(String realSentence) {
        this.realSentence = realSentence;
    }

    public String getRealPhonemes() {
        return realPhonemes;
    }

    public void setRealPhonemes(String realPhonemes) {
        this.realPhonemes = realPhonemes;
    }

    public String getInputShape() {
        return inputShape;
    }

    public void setInputShape(String inputShape) {
        this.inputShape = inputShape;
    }

    public String getLogitsShape() {
        return logitsShape;
    }

    public void setLogitsShape(String logitsShape) {
        this.logitsShape = logitsShape;
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

    public List<Prediction> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<Prediction> predictions) {
        this.predictions = predictions;
    }
}
