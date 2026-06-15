package com.tfg.brain_to_text_web.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.repository.TrialRepository;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;

@ExtendWith(MockitoExtension.class)
class TrialServiceTest {

    @Mock
    private TrialRepository trialRepository;

    @InjectMocks
    private TrialService trialService;

    @Test
    void findSplitsReturnsDistinctSplitsFromRepository() {
        when(trialRepository.findDistinctSplits()).thenReturn(List.of("test", "val"));

        List<String> result = trialService.findSplits();

        assertEquals(List.of("test", "val"), result);
    }

    @Test
    void findTrialsPageUsesAllTrialsWhenSplitIsAll() {
        Page<Trial> expectedPage = new PageImpl<>(List.of(new Trial()));
        when(trialRepository.findAll(any(Pageable.class))).thenReturn(expectedPage);

        Page<Trial> result = trialService.findTrialsPage("all", -3, 8);

        assertEquals(expectedPage, result);

        ArgumentCaptor<Pageable> pageableCaptor = ArgumentCaptor.forClass(Pageable.class);
        verify(trialRepository).findAll(pageableCaptor.capture());
        assertEquals(0, pageableCaptor.getValue().getPageNumber());
        assertEquals(8, pageableCaptor.getValue().getPageSize());
        verify(trialRepository, never()).findBySplit(any(), any());
    }

    @Test
    void findTrialsPageFiltersBySplitWhenSpecificSplitIsProvided() {
        Page<Trial> expectedPage = new PageImpl<>(List.of(new Trial()));
        when(trialRepository.findBySplit(any(), any(Pageable.class))).thenReturn(expectedPage);

        Page<Trial> result = trialService.findTrialsPage("val", 1, 8);

        assertEquals(expectedPage, result);
        ArgumentCaptor<Pageable> pageableCaptor = ArgumentCaptor.forClass(Pageable.class);
        verify(trialRepository).findBySplit(org.mockito.ArgumentMatchers.eq("val"), pageableCaptor.capture());
        assertEquals(1, pageableCaptor.getValue().getPageNumber());
        assertEquals(8, pageableCaptor.getValue().getPageSize());
        assertEquals("sessionName: ASC,split: ASC,trialKey: ASC", pageableCaptor.getValue().getSort().toString());
        verify(trialRepository, never()).findAll(any(Pageable.class));
    }

    @Test
    void createTrialTrimsOptionalValuesAndStoresBlankFieldsAsNull() {
        when(trialRepository.save(any(Trial.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Trial result = trialService.createTrial(
                "t15.2023.08.13",
                "val",
                "trial_0000",
                "  You can see the code.  ",
                "  Y UW  ",
                "  [1, 902, 512]  ",
                " ",
                null,
                "  example notes  "
        );

        assertEquals("t15.2023.08.13", result.getSessionName());
        assertEquals("val", result.getSplit());
        assertEquals("trial_0000", result.getTrialKey());
        assertEquals("You can see the code.", result.getRealSentence());
        assertEquals("Y UW", result.getRealPhonemes());
        assertEquals("[1, 902, 512]", result.getInputShape());
        assertNull(result.getLogitsShape());
        assertNull(result.getSignalImagePath());
        assertEquals("example notes", result.getNotes());
    }

    @Test
    void updateTrialLoadsExistingTrialAndSavesUpdatedValues() {
        Trial existingTrial = new Trial();
        existingTrial.setSessionName("old");
        existingTrial.setSplit("test");
        existingTrial.setTrialKey("trial_old");
        when(trialRepository.findById(5L)).thenReturn(Optional.of(existingTrial));
        when(trialRepository.save(any(Trial.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Trial result = trialService.updateTrial(
                5L,
                "t15.2023.08.18",
                "val",
                "trial_0001",
                "",
                "  HH AW  ",
                null,
                "[223, 41]",
                "/images/signals/example.png",
                ""
        );

        assertEquals("t15.2023.08.18", result.getSessionName());
        assertEquals("val", result.getSplit());
        assertEquals("trial_0001", result.getTrialKey());
        assertNull(result.getRealSentence());
        assertEquals("HH AW", result.getRealPhonemes());
        assertNull(result.getInputShape());
        assertEquals("[223, 41]", result.getLogitsShape());
        assertEquals("/images/signals/example.png", result.getSignalImagePath());
        assertNull(result.getNotes());
    }

    @Test
    void findTrialThrowsExceptionWhenTrialDoesNotExist() {
        when(trialRepository.findById(99L)).thenReturn(Optional.empty());

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> trialService.findTrial(99L)
        );

        assertEquals("Ensayo no encontrado: 99", exception.getMessage());
    }

    @Test
    void findTrialWithPredictionsReturnsTrialWhenItExists() {
        Trial trial = new Trial();
        when(trialRepository.findWithPredictionsById(8L)).thenReturn(Optional.of(trial));

        Trial result = trialService.findTrialWithPredictions(8L);

        assertEquals(trial, result);
    }

    @Test
    void findTrialWithPredictionsThrowsExceptionWhenTrialDoesNotExist() {
        when(trialRepository.findWithPredictionsById(8L)).thenReturn(Optional.empty());

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> trialService.findTrialWithPredictions(8L)
        );

        assertEquals("Ensayo no encontrado: 8", exception.getMessage());
    }

    @Test
    void deleteTrialDelegatesToRepository() {
        trialService.deleteTrial(4L);

        verify(trialRepository).deleteById(4L);
    }

    @Test
    void countTrialsDelegatesToRepository() {
        when(trialRepository.count()).thenReturn(120L);

        long result = trialService.countTrials();

        assertEquals(120L, result);
    }
}
