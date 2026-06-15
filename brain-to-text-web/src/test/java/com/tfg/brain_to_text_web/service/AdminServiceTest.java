package com.tfg.brain_to_text_web.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class AdminServiceTest {

    @Mock
    private AppUserRepository appUserRepository;

    @Mock
    private PredictionRepository predictionRepository;

    @Mock
    private CandidateRepository candidateRepository;

    @Mock
    private TrialService trialService;

    @InjectMocks
    private AdminService adminService;

    @Test
    void authenticateAdminRejectsEmptyCredentialsWithoutQueryingRepository() {
        assertTrue(adminService.authenticateAdmin(" ", "admin123").isEmpty());
        assertTrue(adminService.authenticateAdmin("admin", null).isEmpty());

        verify(appUserRepository, never()).findValidAdmin(any(), any());
    }

    @Test
    void authenticateAdminTrimsUsernameAndDelegatesToRepository() {
        AppUser admin = new AppUser();
        when(appUserRepository.findValidAdmin("admin", "admin123")).thenReturn(Optional.of(admin));

        Optional<AppUser> result = adminService.authenticateAdmin("  admin  ", "admin123");

        assertTrue(result.isPresent());
        assertSame(admin, result.get());
    }

    @Test
    void getStatsAggregatesCountersFromRepositoriesAndTrialService() {
        when(trialService.countTrials()).thenReturn(100L);
        when(predictionRepository.count()).thenReturn(200L);
        when(candidateRepository.count()).thenReturn(300L);
        when(appUserRepository.count()).thenReturn(1L);

        AdminService.AdminStats stats = adminService.getStats();

        assertEquals(100L, stats.trialCount());
        assertEquals(200L, stats.predictionCount());
        assertEquals(300L, stats.candidateCount());
        assertEquals(1L, stats.userCount());
    }

    @Test
    void findPredictionsForTrialDelegatesToRepository() {
        Prediction baseline = new Prediction();
        Prediction proposed = new Prediction();
        when(predictionRepository.findByTrial_IdOrderByModelNameAsc(7L)).thenReturn(List.of(baseline, proposed));

        List<Prediction> result = adminService.findPredictionsForTrial(7L);

        assertEquals(List.of(baseline, proposed), result);
    }

    @Test
    void createPredictionAssociatesTrialTrimsTextAndParsesDecimals() {
        Trial trial = new Trial();
        when(trialService.findTrial(7L)).thenReturn(trial);
        when(predictionRepository.save(any(Prediction.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Prediction result = adminService.createPrediction(
                7L,
                "  baseline  ",
                "  Modelo base  ",
                "  Y UW  ",
                "  yew can see  ",
                " ",
                "0,125",
                "",
                " 0.0748 ",
                "  notes  "
        );

        assertSame(trial, result.getTrial());
        assertEquals("baseline", result.getModelName());
        assertEquals("Modelo base", result.getModelLabel());
        assertEquals("Y UW", result.getPredictedPhonemes());
        assertEquals("yew can see", result.getPredictedText());
        assertNull(result.getPartialText());
        assertEquals(new BigDecimal("0.125"), result.getPerValue());
        assertNull(result.getWerValue());
        assertEquals(new BigDecimal("0.0748"), result.getCheckpointPer());
        assertEquals("notes", result.getNotes());
    }

    @Test
    void createPredictionRequiresModelNameAndModelLabel() {
        when(trialService.findTrial(7L)).thenReturn(new Trial());

        assertThrows(IllegalArgumentException.class, () -> adminService.createPrediction(
                7L,
                " ",
                "Modelo base",
                null,
                null,
                null,
                null,
                null,
                null,
                null
        ));

        verify(predictionRepository, never()).save(any(Prediction.class));
    }

    @Test
    void updatePredictionLoadsExistingPredictionAndUpdatesItsFields() {
        Prediction prediction = new Prediction();
        when(predictionRepository.findById(3L)).thenReturn(Optional.of(prediction));
        when(predictionRepository.save(any(Prediction.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Prediction result = adminService.updatePrediction(
                3L,
                "proposed",
                "Modelo propuesto",
                "K AE N",
                "you can see",
                "partial",
                "0.08",
                "0.20",
                "",
                ""
        );

        assertSame(prediction, result);
        assertEquals("proposed", result.getModelName());
        assertEquals("Modelo propuesto", result.getModelLabel());
        assertEquals("K AE N", result.getPredictedPhonemes());
        assertEquals("you can see", result.getPredictedText());
        assertEquals("partial", result.getPartialText());
        assertEquals(new BigDecimal("0.08"), result.getPerValue());
        assertEquals(new BigDecimal("0.20"), result.getWerValue());
        assertNull(result.getCheckpointPer());
        assertNull(result.getNotes());
    }

    @Test
    void createCandidateAssociatesPredictionAndParsesRank() {
        Prediction prediction = new Prediction();
        when(predictionRepository.findById(4L)).thenReturn(Optional.of(prediction));
        when(candidateRepository.save(any(Candidate.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Candidate result = adminService.createCandidate(4L, " 2 ", "  you can see the code  ");

        assertSame(prediction, result.getPrediction());
        assertEquals(2, result.getRank());
        assertEquals("you can see the code", result.getCandidateText());
    }

    @Test
    void updateCandidateLoadsExistingCandidateAndUpdatesItsFields() {
        Candidate candidate = new Candidate();
        when(candidateRepository.findById(9L)).thenReturn(Optional.of(candidate));
        when(candidateRepository.save(any(Candidate.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Candidate result = adminService.updateCandidate(9L, " 3 ", "  candidate text  ");

        assertSame(candidate, result);
        assertEquals(3, result.getRank());
        assertEquals("candidate text", result.getCandidateText());
    }

    @Test
    void updateCandidateThrowsExceptionWhenCandidateDoesNotExist() {
        when(candidateRepository.findById(404L)).thenReturn(Optional.empty());

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> adminService.updateCandidate(404L, "1", "candidate")
        );

        assertEquals("Candidato no encontrado: 404", exception.getMessage());
        verify(candidateRepository, never()).save(any(Candidate.class));
    }

    @Test
    void createCandidateRequiresExistingPrediction() {
        when(predictionRepository.findById(404L)).thenReturn(Optional.empty());

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> adminService.createCandidate(404L, "1", "candidate")
        );

        assertEquals("Predicción no encontrada: 404", exception.getMessage());
        verify(candidateRepository, never()).save(any(Candidate.class));
    }

    @Test
    void deletePredictionAndDeleteCandidateDelegateToRepositories() {
        adminService.deletePrediction(11L);
        adminService.deleteCandidate(22L);

        verify(predictionRepository).deleteById(11L);
        verify(candidateRepository).deleteById(22L);
    }
}
