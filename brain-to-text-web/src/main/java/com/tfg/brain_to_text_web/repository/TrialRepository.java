package com.tfg.brain_to_text_web.repository;

import com.tfg.brain_to_text_web.model.Trial;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface TrialRepository extends JpaRepository<Trial, Long> {

    Page<Trial> findAll(Pageable pageable);

    Page<Trial> findBySplit(String split, Pageable pageable);

    @EntityGraph(attributePaths = {"predictions"})
    Optional<Trial> findWithPredictionsById(Long id);

    @Query("select distinct t.split from Trial t order by t.split")
    List<String> findDistinctSplits();
}
