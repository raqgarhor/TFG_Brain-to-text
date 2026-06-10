package com.tfg.brain_to_text_web.repository;

import com.tfg.brain_to_text_web.model.Prediction;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.EntityGraph;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    @EntityGraph(attributePaths = {"candidates"})
    List<Prediction> findByTrial_IdOrderByModelNameAsc(Long trialId);
}
